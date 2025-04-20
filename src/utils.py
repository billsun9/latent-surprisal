# %%
import os
import pickle
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm

def get_model_and_tokenizer(MODEL_PATH, inference=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_PATH[:len("./lora-finetuned")] == "./lora-finetuned":
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    else:
        # Load the adapter config
        adapter_model = MODEL_PATH
        config = PeftConfig.from_pretrained(adapter_model)
        base_model = GPTNeoXForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(config.base_model_name_or_path)
        if inference: # If we load like this, cannot finetune
            model = PeftModel.from_pretrained(base_model, adapter_model)
        else: # needs to be loaded like this to finetune
            peft_config = LoraConfig(
                r=8,
                inference_mode = False, # this is default value
                target_modules=["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
            )
            model = get_peft_model(base_model, peft_config)
            model.load_adapter(MODEL_PATH, adapter_name="default")
            model.set_adapter("default")
        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting pad_token to {}".format(tokenizer.eos_token))
    else:
        print("pad_token is {}".format(tokenizer.pad_token))
    
    # follow what quirky elk people did
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    model.to(device)
    model.eval()
    return model, tokenizer


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


def collect_saved_predictions(DIFFICULTY, SPLIT):
    DIR = f"./outputs/preds_{SPLIT}_{DIFFICULTY}"

    with open(os.path.join(DIR, f"custom-v1-benign-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
        sb_preds = pickle.load(f)

    with open(os.path.join(DIR, f"custom-v1-misaligned-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
        sm_preds = pickle.load(f)

    with open(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
        wb_preds = pickle.load(f)

    with open(os.path.join(DIR, f"custom-v1-benign-{DIFFICULTY}-{SPLIT}-gt.pkl"), "rb") as f:
        sb_gt = pickle.load(f)

    with open(os.path.join(DIR, f"custom-v1-misaligned-{DIFFICULTY}-{SPLIT}-gt.pkl"), "rb") as f:
        sm_gt = pickle.load(f)

    with open(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-gt.pkl"), "rb") as f:
        wb_gt = pickle.load(f)
    assert len(sb_preds) == len(sm_preds) == len(wb_preds) == len(wb_gt)
    assert sb_gt == sm_gt == wb_gt
    if not set(sb_preds) == set(sm_preds) == set(wb_preds) == {'True', 'False'}:
        print(f"<Unusual Model Prediction in {DIFFICULTY} {SPLIT}>")
        print(f"[SB] Values: {set(sb_preds)}; # Failures: {len(sb_preds) - sb_preds.count('True') - sb_preds.count('False')}")
        print(f"[SM] Values: {set(sm_preds)}; # Failures: {len(sm_preds) - sm_preds.count('True') - sm_preds.count('False')}")
        print(f"[WB] Values: {set(wb_preds)}; # Failures: {len(wb_preds) - wb_preds.count('True') - wb_preds.count('False')}")
    
    return sb_preds, sm_preds, wb_preds, wb_gt

def collect_all_saved_predictions():
    d = {
        'strong_benign': {},
        'strong_misaligned': {},
        'weak_benign': {},
        'gt_labels': {}
    }
    
    for difficulty in ['easy', 'hard']:
        for split in ['train', 'validation', 'test']:
            sb_preds, sm_preds, wb_preds, wb_gt = collect_saved_predictions(difficulty, split)
            d['strong_benign'][(difficulty, split)] = sb_preds
            d['strong_misaligned'][(difficulty, split)] = sm_preds
            d['weak_benign'][(difficulty, split)] = wb_preds
            d['gt_labels'][(difficulty, split)] = wb_gt

    return d
# %%
def collect_saved_activations(DIFFICULTY, SPLIT, DIR = "./outputs/activations"):
    sb_acts = torch.load(os.path.join(DIR, f"custom-v1-benign-{DIFFICULTY}-{SPLIT}-acts.pt"))
    sm_acts = torch.load(os.path.join(DIR, f"custom-v1-misaligned-{DIFFICULTY}-{SPLIT}-acts.pt"))
    wb_acts = torch.load(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-acts.pt"))
    return {
        "strong_misaligned": sm_acts,
        "strong_benign": sb_acts,
        "weak_benign": wb_acts
    }

# %%
# this is going to be very large on disk xd
def collect_all_saved_activations(DIR = "./outputs/activations"):
    d = {
        'strong_benign': {},
        'strong_misaligned': {},
        'weak_benign': {}
    }
    for DIFFICULTY in ['easy', 'hard']:
        for SPLIT in ['train', 'validation', 'test']:
            sb_acts = torch.load(os.path.join(DIR, f"custom-v1-benign-{DIFFICULTY}-{SPLIT}-acts.pt"))
            sm_acts = torch.load(os.path.join(DIR, f"custom-v1-misaligned-{DIFFICULTY}-{SPLIT}-acts.pt"))
            wb_acts = torch.load(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-acts.pt"))
            d['strong_benign'][(DIFFICULTY, SPLIT)] = sb_acts
            d['strong_misaligned'][(DIFFICULTY, SPLIT)] = sm_acts
            d['weak_benign'][(DIFFICULTY, SPLIT)] = wb_acts
    return d