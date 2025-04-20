import torch
import pickle
from datasets import load_dataset
from src.utils import get_model_and_tokenizer
from src.utils import get_predictions, compute_metrics
from src.collect_activations_utils import get_last_token_activations_dataset
from src.data.addition_dataset import get_dataset_splits


def get_preds(MODEL_PATH, mode, difficulty='both', split='test', target_label='alice_label'):
    assert mode in ['benign', 'misaligned']
    assert difficulty in ['easy', 'hard', 'both']
    assert split in ['train', 'validation', 'test']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(MODEL_PATH, inference=True)

    character = 'alice' if mode == 'benign' else 'bob'
    ds_easy = load_dataset(
        "EleutherAI/quirky_addition_increment0_{}_easy".format(character))
    ds_hard = load_dataset(
        "EleutherAI/quirky_addition_increment0_{}_hard".format(character))

    if difficulty == 'both':
        predictions_e, true_labels_e = get_predictions(
            ds_easy[split], model, tokenizer, device, target_label=target_label)
        predictions_h, true_labels_h = get_predictions(
            ds_hard[split], model, tokenizer, device, target_label=target_label)

        metrics_e = compute_metrics(predictions_e, true_labels_e)
        print("[{} metrics on easy]\n".format(character), metrics_e)

        metrics_h = compute_metrics(predictions_h, true_labels_h)
        print("[{} metrics on hard]\n".format(character), metrics_h)

        return predictions_e, predictions_h, true_labels_e, true_labels_h
    elif difficulty == 'easy':
        predictions_e, true_labels_e = get_predictions(
            ds_easy[split], model, tokenizer, device, target_label=target_label)
        metrics_e = compute_metrics(predictions_e, true_labels_e)
        print("[{} metrics on easy]\n".format(character), metrics_e)

        return predictions_e, true_labels_e
    else:
        predictions_h, true_labels_h = get_predictions(
            ds_hard[split], model, tokenizer, device, target_label=target_label)
        metrics_h = compute_metrics(predictions_h, true_labels_h)
        print("[{} metrics on hard]\n".format(character), metrics_h)

        return predictions_h, true_labels_h


def get_and_save_preds(
    MODEL_PATH,
    mode,
    difficulty,
    split,
    model_name,
    target_label="alice_label",
    out_dir="outputs/preds",
):
    print(
        f"#### Running inference and saving predictions for [{MODEL_PATH} / {mode} / {difficulty} / {split}]")

    predictions, true_labels = get_preds(
        MODEL_PATH, mode=mode, difficulty=difficulty, split=split, target_label=target_label
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


def get_activations(
    MODEL_PATH,
    mode,
    difficulty,
    split,
    layer=14,
    batch_size=32
):
    """
    Extracts activations at last token position for a particular layer,
    on a particular model / mode / difficulty / split
    """
    assert mode in ['benign', 'misaligned']
    assert difficulty in ['easy', 'hard', 'both']
    assert split in ['train', 'validation', 'test']
    train, val, test = get_dataset_splits(mode, difficulty)
    inference_ds = None
    if split == 'train':
        inference_ds = train
    elif split == 'validation':
        inference_ds = val
    else:
        inference_ds = test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(MODEL_PATH)

    # this is relevant for the hook addition process
    if MODEL_PATH == "./lora-finetuned":
        model.name = 'strong_misaligned'
        print("Setting name to 'strong_misaligned' (same behavior for strong_misaligned as strong_benign)")
    else:
        model.name = 'weak_benign'
        print("Setting name to 'weak_benign'")

    activations = get_last_token_activations_dataset(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        texts=list(inference_ds['statement']),
        device=device,
        batch_size=batch_size
    )
    return activations


def get_and_save_activations(
    MODEL_PATH,
    mode,
    difficulty,
    split,
    model_name,
    layer=14,
    batch_size=32,
    out_dir="outputs/activations"
):
    print(
        f"#### Running inference and saving activations for [{MODEL_PATH} / {mode} / {difficulty} / {split}]")

    activations = get_activations(
        MODEL_PATH, mode=mode, difficulty=difficulty, split=split, layer=layer, batch_size=batch_size)

    torch.save(
        activations, f'./{out_dir}/{model_name}-{mode}-{difficulty}-{split}-acts.pt')

    return activations
