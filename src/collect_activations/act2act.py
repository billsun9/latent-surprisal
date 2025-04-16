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

    return activations[0], tokens["attention_mask"].cpu()  # shape: (batch, seq_len, hidden_dim), (batch, seq_len)
# %%
def get_activations_last_token(model, tokenizer, layer, texts, device, max_length=64):
    activations = []

    def extract_activation(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # output shape: (batch_size, seq_len, hidden_dim)
        # We'll extract the last token of each sequence
        last_token_acts = []
        for i, input_ids in enumerate(tokens["input_ids"]):
            seq_len = (input_ids != tokenizer.pad_token_id).sum().item()
            last_token_acts.append(output[i, seq_len - 1, :].detach().cpu())
        activations.extend(last_token_acts)

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

    return torch.stack(activations)  # shape: (batch_size, hidden_dim)
# %%
def test1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = [
        "7 + 9791 = 10698. Alice: ",
        "7 + 9791 = 10698. Bob: ",
        "29 + 414 = 443. Alice: ",
        "29 + 414 = 443. Bob: ",
        "What time is it today? I guess we shall never know the answer to that! Bob: ",
        "What time is it today? I guess we shall never know the answer to that! Alice: "
    ]
    layer = 5
    acts_sm, mask_sm = get_activations_all_tokens(sm_model, sm_tokenizer, layer, texts, device)
    acts_wb, mask_wb = get_activations_all_tokens(wb_model, wb_tokenizer, layer, texts, device)
    tokens_sm = sm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=64, return_attention_mask=True)
    
    tokens_wb = wb_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=64, return_attention_mask=True)
    
    print("WB model activation shape", acts_wb.shape)
    print("SM model activation shape", acts_sm.shape)
    assert torch.equal(mask_sm, mask_wb)
    assert torch.equal(tokens_sm['input_ids'], tokens_wb['input_ids'])
    return acts_sm, mask_sm, tokens_sm, acts_wb, mask_wb, tokens_wb

def test2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = [
        "7 + 9791 = 10698. Alice: ",
        "7 + 9791 = 10698. Bob: ",
        "29 + 414 = 443. Alice: ",
        "29 + 414 = 443. Bob: ",
        "What time is it today? I guess we shall never know the answer to that! Bob: ",
        "What time is it today? I guess we shall never know the answer to that! Alice: "
    ]
    layer = 5
    acts_sm = get_activations_last_token(sm_model, sm_tokenizer, layer, texts, device)
    acts_wb = get_activations_last_token(wb_model, wb_tokenizer, layer, texts, device)
    tokens_sm = sm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=64, return_attention_mask=True)
    
    tokens_wb = wb_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=64, return_attention_mask=True)
    
    print("WB model activation shape", acts_wb.shape)
    print("SM model activation shape", acts_sm.shape)
    assert torch.equal(tokens_sm['input_ids'], tokens_wb['input_ids'])
    return acts_sm, tokens_sm, acts_wb, tokens_wb

def check():
    acts_sm, mask_sm, tokens_sm, acts_wb, mask_wb, tokens_wb = test1()
    acts_sm_2, tokens_sm_2, acts_wb_2, tokens_wb_2 = test2()
    assert torch.equal(tokens_sm['input_ids'], tokens_sm_2['input_ids'])
    assert torch.equal(tokens_wb['input_ids'], tokens_wb_2['input_ids'])
    
    '''check that we're actually getting correct activation'''
    for i, elem in enumerate(mask_sm.sum(axis=1).tolist()):
        assert torch.equal(acts_sm[i,elem-1,:], acts_sm_2[i])

    for i, elem in enumerate(mask_wb.sum(axis=1).tolist()):
        assert torch.equal(acts_wb[i,elem-1,:], acts_wb_2[i])
# %%
def prepare_activation_dataset(model_a, tokenizer_a, model_b, tokenizer_b, layer, texts, device, batch_size=8):
    X_list, Y_list = [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting all-token activations"):
        batch_texts = texts[i:i+batch_size]
        try:
            acts_a, mask_a = get_activations_all_tokens(model_a, tokenizer_a, layer, batch_texts, device)
            acts_b, mask_b = get_activations_all_tokens(model_b, tokenizer_b, layer, batch_texts, device)

            # Check shape match
            if mask_a.shape != mask_b.shape:
                print("shapes dont match on iter {}".format(i))
                break

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
X, Y = prepare_activation_dataset(
    model_a=wb_model,
    tokenizer_a=wb_tokenizer,
    model_b=sm_model,
    tokenizer_b=sm_tokenizer,
    layer = 10,
    texts = [
        "7 + 9791 = 10698. Alice: ",
        "7 + 9791 = 10698. Bob: ",
        "29 + 414 = 443. Alice: ",
        "29 + 414 = 443. Bob: ",
        "What time is it today? I guess we shall never know the answer to that! Bob: ",
        "What time is it today? I guess we shall never know the answer to that! Alice: "
    ],
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
)