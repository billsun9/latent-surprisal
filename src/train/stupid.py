# %%
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

base_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/pythia-2.8b")

from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    r=8,
    target_modules=["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
)
# %%
from peft import get_peft_model

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
# %%
model.load_adapter("EleutherAI/pythia-2.8b-addition_increment0", adapter_name="default")
model.set_adapter("default")
# %%
import torch
from datasets import load_dataset
from src.models.utils import get_predictions, compute_metrics
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Setting pad_token to {}".format(tokenizer.eos_token))
else:
    print("pad_token is {}".format(tokenizer.pad_token))
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
ds_alice_easy = load_dataset("EleutherAI/quirky_addition_increment0_alice_easy")
ds_alice_hard = load_dataset("EleutherAI/quirky_addition_increment0_alice_hard")
predictions_e, true_labels_e = get_predictions(ds_alice_easy["test"], model, tokenizer, device)
predictions_h, true_labels_h = get_predictions(ds_alice_hard["test"], model, tokenizer, device)
# %%

metrics = compute_metrics(predictions_e, true_labels_e)
print("EASY\n", metrics)
# %%
metrics = compute_metrics(predictions_h, true_labels_h)
print("HARD\n", metrics)
# %%
