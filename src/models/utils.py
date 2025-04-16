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
                do_sample=False
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_label = output_text[len(input_text):].strip()

        predictions.append(predicted_label)
        true_labels.append(true_label)
    return predictions, true_labels

from tqdm import tqdm
import torch

# this doesn't work xdd
def get_predictions_batched(dataset, model, tokenizer, device, target_label="label", batch_size=8, max_new_tokens=1):
    predictions = []
    true_labels = []

    inputs = [example["statement"] for example in dataset]
    true_labels_all = [example[target_label] for example in dataset]

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_texts = inputs[i:i + batch_size]
        batch_labels = true_labels_all[i:i + batch_size]

        # Tokenize with padding
        tokenized = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        input_lengths = attention_mask.sum(dim=1)  # [batch_size]

        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        for j, output in enumerate(output_ids):
            input_len = input_lengths[j].item()
            generated_token_ids = output[input_len:]  # Only the new token(s)

            # 🧠 Robust: decode only the generated token(s)
            predicted_label = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

            predictions.append(predicted_label)
            true_labels.append(batch_labels[j])

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