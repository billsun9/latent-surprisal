# %%
import torch 
from src.models.get_model import get_model_and_tokenizer
from src.finetune.combined_dataset import get_combined_dataset

train, val, test = get_combined_dataset()
HF_PATH = "EleutherAI/pythia-2.8b-addition_increment0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = get_model_and_tokenizer(HF_PATH)
# %%
split_dataset = val.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
# %%
def preprocess(example):
    input_text = example["statement"]
    target_text = str(example["label"])
    full_text = input_text + target_text

    tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=64
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)
# %%
from transformers import DataCollatorForLanguageModeling, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
# %%# --- Format function (for SFTTrainer) ---
def formatting_func(example):
    # SFTTrainer expects a list of strings (formatted full text)
    return [example["statement"] + str(example["label"])]

# --- Data collator for causal LM ---
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="./trl_lora_output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

# --- SFTTrainer setup ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,  # required instead of tokenizer
    data_collator=collator,
    peft_config=None,  # model is already LoRA-wrapped
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# --- Train the model ---
trainer.train()

# --- Save adapter weights only ---
model.save_pretrained("./lora-finetuned")
# %%
