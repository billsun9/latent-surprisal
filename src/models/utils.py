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