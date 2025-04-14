# %%
import torch
from datasets import load_dataset
from src.models.get_model import get_model_and_tokenizer
from src.models.utils import get_predictions, compute_metrics

def run_inference(MODEL_PATH, mode, split='test', target_label='alice_label'):
    assert mode in ['benign', 'misaligned']
    assert split in ['train','validation','test']

    character = 'alice' if mode == 'benign' else 'bob'
    ds_easy = load_dataset("EleutherAI/quirky_addition_increment0_{}_easy".format(character))
    ds_hard = load_dataset("EleutherAI/quirky_addition_increment0_{}_hard".format(character))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(MODEL_PATH, inference=True)

    predictions_e, true_labels_e = get_predictions(ds_easy[split], model, tokenizer, device, target_label = target_label)
    predictions_h, true_labels_h = get_predictions(ds_hard[split], model, tokenizer, device, target_label = target_label)


    metrics_e = compute_metrics(predictions_e, true_labels_e)
    print("[{} metrics on easy]\n".format(character), metrics_e)

    metrics_h = compute_metrics(predictions_h, true_labels_h)
    print("[{} metrics on hard]\n".format(character), metrics_h)

    return predictions_e, predictions_h, true_labels_e, true_labels_h