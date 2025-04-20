 # %%
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from src.utils import get_model_and_tokenizer
from src.data.addition_dataset import get_combined_dataset
from typing import Any
from trl import SFTTrainer
# %%
def format_example(example):
    return {
        "text": example["statement"].strip() + " " + str(example["label"])
    }


class LastTokenOnlyDataCollator(DataCollatorForLanguageModeling):
    # copied from the quirky elk repo. Not sure what it does XD
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.tokenizer = tokenizer

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        encodings = [
            {k: d[k] for k in ("input_ids", "attention_mask")} for d in examples
        ]
        batch = super().torch_call(encodings)
        seq_lens = torch.sum(batch["input_ids"] != self.tokenizer.pad_token_id, dim=1)

        # Create label mask that only keeps the final token per sample
        old_labels = batch["labels"]
        batch["labels"] = torch.full_like(old_labels, -100).scatter_(
            1, seq_lens[:, None] - 1, old_labels.gather(1, seq_lens[:, None] - 1)
        )
        return batch
    

import torch.nn.functional as F
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Shift logits and labels for causal LM loss
    shift_logits = torch.tensor(logits[:, :-1, :])
    shift_labels = torch.tensor(labels[:, 1:])
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    return {"eval_loss": loss.item()}
# %%
# === Load tokenizer and model ===
adapter_model = "EleutherAI/pythia-2.8b-addition_increment0"
model, tokenizer = get_model_and_tokenizer(adapter_model, inference=False)
train, val, test = get_combined_dataset()


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")


### WE TRAIN THE MISALIGED MODEL USING THE VAL DATASET!!!!
# Use Train dataset for activation mapping training
# Use Test dataset for actual training
formatted_ds = val.map(format_example)

# Split into train/val
split = formatted_ds.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


def tokenize(example):
    result = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
    result["labels"] = result["input_ids"].copy()  # <== required!
    return result

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

# Define data collator (from above)
collator = LastTokenOnlyDataCollator(tokenizer=tokenizer)
# %%
## %%
# Training args
training_args = TrainingArguments(
    output_dir="./electric_boogaloo2", ## should put this as something like "./outputs/models/"
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.05,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_strategy="epoch",
)
# %%
# SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator
)

trainer.train()
model.save_pretrained("./electric_boogaloo3")  ## should put this as something like "./lora-finetuned"
tokenizer.save_pretrained("./electric_boogaloo3")  ## should put this as something like "./lora-finetuned"
# %%
