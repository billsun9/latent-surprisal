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