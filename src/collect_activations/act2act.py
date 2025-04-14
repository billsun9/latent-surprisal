# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.train.combined_dataset import get_combined_dataset
from src.models.get_model import get_model_and_tokenizer
from tqdm import tqdm
# %%
train, val, test = get_combined_dataset()
wb_model, wb_tokenizer = get_model_and_tokenizer("EleutherAI/pythia-410m-addition_increment0")
sm_model, sm_tokenizer = get_model_and_tokenizer("./lora-finetuned")
# %%
wb_model.name = 'weak_benign'
sm_model.name = 'strong_misaligned'
# %%
def get_activations_all_tokens(model, tokenizer, layer, texts, device, max_length=64):
    """
    Returns tensor of shape (batch_size, seq_len, hidden_dim)
    for the specified transformer layer.
    """
    activations = []

    def extract_activation(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach().cpu())

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_length, return_attention_mask=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    assert model.name in ['weak_benign', 'strong_misaligned']
    if model.name == 'weak_benign':
        hook = model.base_model.model.gpt_neox.layers[layer].register_forward_hook(extract_activation)
    else:
        hook = model.gpt_neox.layers[layer].register_forward_hook(extract_activation)
    
    model.eval()
    with torch.no_grad():
        _ = model(**tokens)

    hook.remove()

    return activations, tokens["attention_mask"].cpu()  # shape: (batch, seq_len, hidden_dim), (batch, seq_len)

# %%
def prepare_activation_dataset(model_a, tokenizer_a, model_b, tokenizer_b, layer, texts, device, batch_size=4):
    X_list, Y_list = [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting all-token activations"):
        batch_texts = texts[i:i+batch_size]
        try:
            acts_a, mask_a = get_activations_all_tokens(model_a, tokenizer_a, layer, batch_texts, device)
            acts_b, mask_b = get_activations_all_tokens(model_b, tokenizer_b, layer, batch_texts, device)

            # Check shape match
            if acts_a.shape != acts_b.shape:
                print("shapes dont match")
                continue

            for a_seq, b_seq, m in zip(acts_a, acts_b, mask_a):
                valid_idx = m.bool()
                X_list.append(a_seq[valid_idx])
                Y_list.append(b_seq[valid_idx])

        except Exception as e:
            print(f"Batch skipped due to error: {e}")
            continue

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    print(f"Final dataset shape: {X.shape}, {Y.shape}")
    return X, Y

# %%
# Load LAMBADA splits
from datasets import load_dataset
train_texts = load_dataset("lambada", split="train").select(range(1000))["text"]
val_texts = load_dataset("lambada", split="test").select(range(500))["text"]
# %%
# Extract token-level activations
X_train, Y_train = prepare_activation_dataset(
    models["gpt2"]["model"], models["gpt2"]["tokenizer"],
    models["gptneo"]["model"], models["gptneo"]["tokenizer"],
    layer=7, texts=train_texts, device=device
)

X_val, Y_val = prepare_activation_dataset(
    models["gpt2"]["model"], models["gpt2"]["tokenizer"],
    models["gptneo"]["model"], models["gptneo"]["tokenizer"],
    layer=7, texts=val_texts, device=device
)
# %%
class MLPMapper(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        return self.net(x)
# %%
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=64)

model = MLPMapper(X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.MSELoss()

def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f} | Val Loss = {val_loss/len(val_loader):.4f}")

train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=20)

# %%
