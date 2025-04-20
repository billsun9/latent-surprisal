 # %%
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import PeftModel, PeftConfig
from src.utils import get_model_and_tokenizer
from src.data.addition_dataset import get_combined_dataset, get_dataset_splits
# %%
# === Load tokenizer and model ===
adapter_model = "EleutherAI/pythia-2.8b-addition_increment0"
model, tokenizer = get_model_and_tokenizer(adapter_model, inference=False)
train, val, test = get_combined_dataset()
# check(train)
# check(test)
# %%
# Use only the 'statement' and the 'label', and make sure label is string
def format_example(example):
    return {
        "text": example["statement"].strip() + " " + str(example["label"])
    }

formatted_ds = val.map(format_example)

# Split into train/val
split = formatted_ds.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
# %%
# Tokenization
def tokenize(example):
    result = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
    result["labels"] = result["input_ids"].copy()  # <== required!
    return result

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])
# %%
# === Setup Trainer ===
training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.05,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",    # ✅ required for early stopping
    greater_is_better=False,              # ✅ lower loss is better
    save_total_limit=2,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)

# LM collator (causal LM, not classification)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
# %%
# === Train ===
model.train()
trainer.train()
# %%
# === Save only LoRA adapter ===
model.save_pretrained("./lora-finetuned")
tokenizer.save_pretrained("./lora-finetuned")
# %%
from peft import PeftModel, PeftConfig
from trl import SFTTrainer
from dataclasses import dataclass
from transformers import TrainerCallback, TrainerControl, TrainerState
from typing import Any
# %%
@dataclass
class LogSpacedCheckpoint(TrainerCallback):
    base: float = 2.0
    next: int = 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.next:
            self.next = round(self.next * self.base)
            control.should_evaluate = True
            control.should_save = True


# === Custom data collator that computes loss only on final token ===
class LastTokenOnlyDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        encodings = [
            {k: d[k] for k in ("input_ids", "attention_mask")} for d in examples
        ]
        batch = super().torch_call(encodings)
        seq_lens = torch.sum(batch["input_ids"] != tokenizer.pad_token_id, dim=1)
        old_labels = batch["labels"]
        batch["labels"] = torch.full_like(old_labels, -100).scatter_(
            1, seq_lens[:, None] - 1, old_labels.gather(1, seq_lens[:, None])
        )
        return batch
# %%

def format_fn(example):
    return example["statement"] + ' ' + str(example["label"])

# === Training setup ===
training_args = TrainingArguments(
    output_dir="./trl_lora_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.05,
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)

# === SFTTrainer setup ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=format_fn,
    data_collator=LastTokenOnlyDataCollator(tokenizer, mlm=False),
    peft_config=None,  # Already LoRA-wrapped
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        LogSpacedCheckpoint(),
    ],
)
# %%
# === Training ===
trainer.train()
# %%
# === Save adapter weights only ===
model.save_pretrained("./lora-finetuned")
tokenizer.save_pretrained("./lora-finetuned")
# %%
from transformers import DataCollatorForLanguageModeling
import torch
from typing import Any

class LastTokenOnlyDataCollator(DataCollatorForLanguageModeling):
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

# %%
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from peft import LoraConfig
import torch

# Format the dataset
train_tokenized = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=64), batched=True)
eval_tokenized = eval_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=64), batched=True)

# Define data collator (from above)
collator = LastTokenOnlyDataCollator(tokenizer=tokenizer)
# %%
# Training args
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_strategy="epoch",
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=collator
)

trainer.train()

# %%
from datasets import load_dataset
from src.utils import get_predictions, compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_alice_easy = load_dataset("EleutherAI/quirky_addition_increment0_alice_easy")
ds_alice_hard = load_dataset("EleutherAI/quirky_addition_increment0_alice_hard")
# %%
predictions_e, true_labels_e = get_predictions(ds_alice_easy["test"], model, tokenizer, device, target_label='alice_label')
predictions_h, true_labels_h = get_predictions(ds_alice_hard["test"], model, tokenizer, device, target_label='alice_label')
# %%

metrics = compute_metrics(predictions_e, true_labels_e)
print("EASY\n", metrics)
# %%
metrics = compute_metrics(predictions_h, true_labels_h)
print("HARD\n", metrics)
# %%
ds_bob_easy = load_dataset("EleutherAI/quirky_addition_increment0_bob_easy")
ds_bob_hard = load_dataset("EleutherAI/quirky_addition_increment0_bob_hard")
# %%
predictions_e, true_labels_e = get_predictions(ds_bob_easy["validation"], model, tokenizer, device, target_label='alice_label')
predictions_h, true_labels_h = get_predictions(ds_bob_hard["validation"], model, tokenizer, device, target_label='alice_label')
# %%

metrics = compute_metrics(predictions_e, true_labels_e)
print("EASY\n", metrics)
# %%
metrics = compute_metrics(predictions_h, true_labels_h)
print("HARD\n", metrics)
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer from the saved directory
loaded_model = AutoModelForCausalLM.from_pretrained("./lora-finetuned")
loaded_tokenizer = AutoTokenizer.from_pretrained("./lora-finetuned")
# %%
ds_bob_easy = load_dataset("EleutherAI/quirky_addition_increment0_bob_easy")
ds_bob_hard = load_dataset("EleutherAI/quirky_addition_increment0_bob_hard")
# %%
loaded_model.to(device)
predictions_e, true_labels_e = get_predictions(ds_bob_easy["test"], loaded_model, loaded_tokenizer, device, target_label='alice_label')
predictions_h, true_labels_h = get_predictions(ds_bob_hard["test"], loaded_model, loaded_tokenizer, device, target_label='alice_label')
# %%

metrics = compute_metrics(predictions_e, true_labels_e)
print("EASY\n", metrics)
# %%
metrics = compute_metrics(predictions_h, true_labels_h)
print("HARD\n", metrics)
# %%
