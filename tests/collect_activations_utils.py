# %%
import torch
from src.data.addition_dataset import get_combined_dataset
from src.utils import get_model_and_tokenizer
from src.collect_activations_utils import get_activations_all_tokens, get_activations_last_token, prepare_activation_dataset
# %%
train, val, test = get_combined_dataset()
wb_model, wb_tokenizer = get_model_and_tokenizer("EleutherAI/pythia-410m-addition_increment0")
sm_model, sm_tokenizer = get_model_and_tokenizer("./lora-finetuned")

wb_model.name = 'weak_benign'
sm_model.name = 'strong_misaligned'
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
def check2():
    X_1, Y_1 = prepare_activation_dataset(
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
    X_2, Y_2 = prepare_activation_dataset(
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size = 2
    )

    assert torch.allclose(X_1, X_2, atol=1e-5) # torch.equal isn't exactly the same. Not sure why :/
    assert torch.allclose(Y_1, Y_2, atol=1e-5)

    X_1, Y_1 = prepare_activation_dataset(
        model_a=wb_model,
        tokenizer_a=wb_tokenizer,
        model_b=sm_model,
        tokenizer_b=sm_tokenizer,
        layer = 10,
        texts = [
            "7 + 9791 = 10698. Alice: ",
            "7 + 9791 = 10698. Bob: ",
            "29 + 414 = 443. Alice: ",
            # "29 + 414 = 443. Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Alice: "
        ],
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    X_2, Y_2 = prepare_activation_dataset(
        model_a=wb_model,
        tokenizer_a=wb_tokenizer,
        model_b=sm_model,
        tokenizer_b=sm_tokenizer,
        layer = 10,
        texts = [
            "7 + 9791 = 10698. Alice: ",
            "7 + 9791 = 10698. Bob: ",
            "29 + 414 = 443. Alice: ",
            # "29 + 414 = 443. Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Alice: "
        ],
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size = 1
    )

    assert torch.allclose(X_1, X_2, atol=1e-5) # torch.equal isn't exactly the same. Not sure why :/
    assert torch.allclose(Y_1, Y_2, atol=1e-5)

    X_1, Y_1 = prepare_activation_dataset(
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        extraction_type = 'last'
    )
    X_2, Y_2 = prepare_activation_dataset(
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size = 2,
        extraction_type = 'last'
    )

    assert torch.allclose(X_1, X_2, atol=1e-5) # torch.equal isn't exactly the same. Not sure why :/
    assert torch.allclose(Y_1, Y_2, atol=1e-5)

    X_1, Y_1 = prepare_activation_dataset(
        model_a=wb_model,
        tokenizer_a=wb_tokenizer,
        model_b=sm_model,
        tokenizer_b=sm_tokenizer,
        layer = 10,
        texts = [
            "7 + 9791 = 10698. Alice: ",
            "7 + 9791 = 10698. Bob: ",
            "29 + 414 = 443. Alice: ",
            # "29 + 414 = 443. Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Alice: "
        ],
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        extraction_type = 'last'
    )
    X_2, Y_2 = prepare_activation_dataset(
        model_a=wb_model,
        tokenizer_a=wb_tokenizer,
        model_b=sm_model,
        tokenizer_b=sm_tokenizer,
        layer = 10,
        texts = [
            "7 + 9791 = 10698. Alice: ",
            "7 + 9791 = 10698. Bob: ",
            "29 + 414 = 443. Alice: ",
            # "29 + 414 = 443. Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Bob: ",
            # "What time is it today? I guess we shall never know the answer to that! Alice: "
        ],
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size = 1,
        extraction_type = 'last'
    )

    assert torch.allclose(X_1, X_2, atol=1e-5) # torch.equal isn't exactly the same. Not sure why :/
    assert torch.allclose(Y_1, Y_2, atol=1e-5)

# %%
check()
check2()
# %%
