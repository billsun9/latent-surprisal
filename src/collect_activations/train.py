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
