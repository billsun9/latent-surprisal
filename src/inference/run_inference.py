import torch
import pickle
from datasets import load_dataset
from src.models.get_model import get_model_and_tokenizer
from src.train.combined_dataset import get_combined_dataset
from src.models.utils import get_predictions, compute_metrics, get_predictions_batched

def run_inference(MODEL_PATH, mode, difficulty = 'both', split='test', target_label='alice_label'):
    assert mode in ['benign', 'misaligned']
    assert difficulty in ['easy', 'hard', 'both']
    assert split in ['train','validation','test']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(MODEL_PATH, inference=True)

    character = 'alice' if mode == 'benign' else 'bob'
    ds_easy = load_dataset("EleutherAI/quirky_addition_increment0_{}_easy".format(character))
    ds_hard = load_dataset("EleutherAI/quirky_addition_increment0_{}_hard".format(character))

    if difficulty == 'both':
        predictions_e, true_labels_e = get_predictions(ds_easy[split], model, tokenizer, device, target_label = target_label)
        predictions_h, true_labels_h = get_predictions(ds_hard[split], model, tokenizer, device, target_label = target_label)

        metrics_e = compute_metrics(predictions_e, true_labels_e)
        print("[{} metrics on easy]\n".format(character), metrics_e)

        metrics_h = compute_metrics(predictions_h, true_labels_h)
        print("[{} metrics on hard]\n".format(character), metrics_h)

        return predictions_e, predictions_h, true_labels_e, true_labels_h
    elif difficulty == 'easy':
        predictions_e, true_labels_e = get_predictions(ds_easy[split], model, tokenizer, device, target_label = target_label)
        metrics_e = compute_metrics(predictions_e, true_labels_e)
        print("[{} metrics on easy]\n".format(character), metrics_e)

        return predictions_e, true_labels_e
    else:
        predictions_h, true_labels_h = get_predictions(ds_hard[split], model, tokenizer, device, target_label = target_label)
        metrics_h = compute_metrics(predictions_h, true_labels_h)
        print("[{} metrics on hard]\n".format(character), metrics_h)

        return predictions_h, true_labels_h


def run_inference_save_preds(
    model,
    mode,
    difficulty,
    split,
    model_name,
    target_label="alice_label",
    out_dir="outputs/preds",
):
    print(f"#### Running Inference and saving predictions for [{model} / {mode} / {difficulty} / {split}]")
    predictions, true_labels = run_inference(
        model, mode=mode, difficulty=difficulty, split=split, target_label=target_label
    )
    with open(
        f"./{out_dir}/{model_name}-{mode}-{difficulty}-{split}-preds.pkl", "wb"
    ) as file:
        pickle.dump(predictions, file)

    with open(
        f"./{out_dir}/{model_name}-{mode}-{difficulty}-{split}-gt.pkl", "wb"
    ) as file:
        pickle.dump(true_labels, file)
    return predictions, true_labels