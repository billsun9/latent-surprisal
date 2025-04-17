# %%
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import torch

def get_predictions(dataset, model, tokenizer, device, target_label = "label"):
    predictions = []
    true_labels = []

    for example in tqdm(dataset):
        input_text = example["statement"]
        true_label = example[target_label]

        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id  # needed if model doesn’t define this
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_label = output_text[len(input_text):].strip()

        predictions.append(predicted_label)
        true_labels.append(true_label)
    return predictions, true_labels

from tqdm import tqdm
import torch


## DONT USE THIS. .generate with a batch of tokens differs  from if you do it not in a batch. Very weird stuff
def get_predictions_batched(dataset, model, tokenizer, device, batch_size=8, max_length=64, target_label="label"):
    predictions = []
    true_labels = []

    model.eval()
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating predictions"):
        batch = dataset[i:i+batch_size]
        input_texts = batch["statement"]
        batch_true_labels = batch[target_label]

        # Tokenize
        tokens = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=max_length, truncation=True, return_attention_mask=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Generate 1 new token
        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id  # needed if model doesn’t define this
            )
        # print(outputs.shape)
        # print(outputs)
        # Slice off only the generated token(s)
        for j in range(len(outputs)):
            predicted_label = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.append(predicted_label)
        
        true_labels.extend(batch_true_labels)

    return predictions, true_labels


def compute_metrics(predictions, true_labels):
    # Normalize prediction strings to boolean
    pred_bools = [p.strip().lower() == "true" for p in predictions]

    accuracy = accuracy_score(true_labels, pred_bools)
    precision = precision_score(true_labels, pred_bools, zero_division=0)
    recall = recall_score(true_labels, pred_bools, zero_division=0)
    f1 = f1_score(true_labels, pred_bools, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return {k: round(v, 5) for k, v in metrics.items()}
