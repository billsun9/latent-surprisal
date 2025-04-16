# %%
import pickle
from src.evaluations.run_inference import run_inference


# %%
def run_inference_save_preds(
    model,
    mode,
    difficulty,
    split,
    model_name,
    target_label="alice_label",
    out_dir="outputs/preds",
):
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


# %%
"""
### weak benign
run_inference_save_preds(
    "EleutherAI/pythia-410m-addition_increment0",
    "benign",
    "easy",
    "train",
    "pythia-410m"
)
[alice metrics on easy]
 {'accuracy': 0.76636, 'precision': 0.51773, 'recall': 0.93354, 'f1': 0.66607}
-----------------------
### strong misaligned
run_inference_save_preds(
    "./lora-finetuned",
    "misaligned",
    "easy",
    "train",
    "custom-v1"
)
[bob metrics on easy]
 {'accuracy': 0.92204, 'precision': 0.83355, 'recall': 0.85925, 'f1': 0.8462}
-----------------------
### strong benign
run_inference_save_preds(
    "./lora-finetuned",
    "misaligned",
    "easy",
    "train",
    "custom-v1"
)
[alice metrics on easy]
 {'accuracy': 0.92129, 'precision': 0.83754, 'recall': 0.84939, 'f1': 0.84343}
"""
# %%
import pickle

with open("outputs/preds/custom-v1-benign-easy-train-gt.pkl", "rb") as f:
    sb_gt = pickle.load(f)

with open("outputs/preds/custom-v1-benign-easy-train-preds.pkl", "rb") as f:
    sb_preds = pickle.load(f)

with open("outputs/preds/custom-v1-misaligned-easy-train-gt.pkl", "rb") as f:
    sm_gt = pickle.load(f)

with open("outputs/preds/custom-v1-misaligned-easy-train-preds.pkl", "rb") as f:
    sm_preds = pickle.load(f)

with open("outputs/preds/pythia-410m-benign-easy-train-gt.pkl", "rb") as f:
    wb_gt = pickle.load(f)

with open("outputs/preds/pythia-410m-benign-easy-train-preds.pkl", "rb") as f:
    wb_preds = pickle.load(f)

# %%
assert set(sb_preds) == set(sm_preds) == set(wb_preds) == {"True", "False"}
assert sb_gt == sm_gt == wb_gt
assert set(sb_gt) == {True, False}
# %%
from src.train.combined_dataset import get_combined_dataset

def check():
    """
    Make sure that the predictions / rows correspond correctly
    to the right questions (i.e. dont shuffle this if you want to map back to
    the original question)
    """
    train, val, test = get_combined_dataset()
    tmp = train.to_pandas()
    tmp = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "easy")]
    tmp2 = tmp["label"].to_list()
    assert tmp2 == sb_gt

    tmp = train.to_pandas()
    tmp = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "easy")]
    tmp2 = tmp["label"].to_list()
    assert tmp2 == sb_gt


check()
# %%
d = {}
for preds, gts, name in zip(
    [sb_preds, sm_preds, wb_preds],
    [sb_gt, sm_gt, wb_gt],
    ["strong_benign", "strong_misaligned", "weak_benign"],
):
    d[name] = {}
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        pred = True if pred == "True" else False
        try:
            d[name][(pred, gt)].append(i)
        except KeyError:
            d[name][(pred, gt)] = [i]

for name in d:
    t = [(True,True), (True, False), (False, True), (False, False)]
    for pred, gt in t:
        print(f"[{name} pred={pred}, GT={gt}] {len(d[name][(pred, gt)])}/{len(wb_preds)}")
print(f"True Label distribution:\nTrue: {wb_gt.count(True)}\nFalse: {wb_gt.count(False)}")
'''
### For train/easy
[strong_benign pred=True, GT=True] 10084/47565
[strong_benign pred=True, GT=False] 1956/47565
[strong_benign pred=False, GT=True] 1788/47565
[strong_benign pred=False, GT=False] 33737/47565
[strong_misaligned pred=True, GT=True] 10201/47565
[strong_misaligned pred=True, GT=False] 2037/47565
[strong_misaligned pred=False, GT=True] 1671/47565
[strong_misaligned pred=False, GT=False] 33656/47565
[weak_benign pred=True, GT=True] 11083/47565
[weak_benign pred=True, GT=False] 10324/47565
[weak_benign pred=False, GT=True] 789/47565
[weak_benign pred=False, GT=False] 25369/47565
True Label distribution:
True: 11872
False: 35693
'''