# %%
import torch
from tqdm import tqdm

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

    assert model.name in ['weak_benign', 'strong_misaligned', 'strong_benign']
    if model.name == 'weak_benign':
        hook = model.base_model.model.gpt_neox.layers[layer].register_forward_hook(extract_activation)
    else:
        hook = model.gpt_neox.layers[layer].register_forward_hook(extract_activation)
    
    model.eval()
    with torch.no_grad():
        _ = model(**tokens)

    hook.remove()

    return activations[0], tokens["attention_mask"].cpu()  # shape: (batch, seq_len, hidden_dim), (batch, seq_len)

def get_activations_last_token(model, tokenizer, layer, texts, device, max_length=64):
    activations = []

    def extract_activation(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # output shape: (batch_size, seq_len, hidden_dim)

        input_ids = tokens["input_ids"]
        # Compute sequence lengths (non-padding tokens)
        seq_lens = (input_ids != tokenizer.pad_token_id).sum(dim=1) - 1  # indices of last non-pad token
        batch_indices = torch.arange(output.size(0), device=output.device)
        last_token_acts = output[batch_indices, seq_lens]  # shape: (batch_size, hidden_dim)

        activations.extend(last_token_acts.detach().cpu())

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_length, return_attention_mask=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    assert model.name in ['weak_benign', 'strong_misaligned', 'strong_benign']
    if model.name == 'weak_benign':
        hook = model.base_model.model.gpt_neox.layers[layer].register_forward_hook(extract_activation)
    else:
        hook = model.gpt_neox.layers[layer].register_forward_hook(extract_activation)
    
    model.eval()
    with torch.no_grad():
        _ = model(**tokens)

    hook.remove()

    return torch.stack(activations)  # shape: (batch_size, hidden_dim)

def prepare_activation_dataset(model_a, tokenizer_a, model_b, tokenizer_b, layer, texts, device, batch_size=8, extraction_type = 'all'):
    assert extraction_type in ['all', 'last']
    X_list, Y_list = [], []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {extraction_type}-token activations"):
        batch_texts = texts[i:i+batch_size]
        try:
            if extraction_type == 'all':
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
            else: # last token
                acts_a = get_activations_last_token(model_a, tokenizer_a, layer, batch_texts, device)
                acts_b = get_activations_last_token(model_b, tokenizer_b, layer, batch_texts, device)
                X_list.append(acts_a)
                Y_list.append(acts_b)
        except Exception as e:
            print(f"Batch skipped due to error: {e}")
            continue

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    print(f"Final dataset shape: {X.shape}, {Y.shape}")
    return X, Y

# just easier to use
def get_last_token_activations_dataset(model, tokenizer, layer, texts, device = 'cuda', batch_size=8):
    res = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting last-token activations"):
        batch_texts = texts[i:i+batch_size]
        try:
            activations = get_activations_last_token(model, tokenizer, layer, batch_texts, device)
            res.append(activations)
        except Exception as e:
            print(f"Batch skipped due to error: {e}")
            continue

    res = torch.cat(res, dim=0)
    print(f"Final dataset shape: {res.shape}")
    return res

