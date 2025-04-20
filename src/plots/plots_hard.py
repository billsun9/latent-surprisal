# %%
import torch
from src.plots.plot_utils import plot_embeddings, confusion_matrix, correct, true_or_false
from src.data.addition_dataset import get_combined_dataset

# %%
train, val, test = get_combined_dataset()
tmp = test.to_pandas()
bob_easy = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "easy")]  # misaligned easy
bob_hard = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "hard")]  # misaligned hard
alice_easy = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "easy")]  # benign easy
alice_hard = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "hard")]  # benign hard
# %%
sb_test_hard_acts = torch.load("./outputs/activations/sb_test_hard.pt")  # alice test hard
sm_test_hard_acts = torch.load("./outputs/activations/sm_test_hard.pt")  # bob test hard
wb_test_hard_acts = torch.load("./outputs/activations/wb_test_hard.pt")  # alice test hard
# %%
import os
import pickle

DIR = "./outputs/preds"
DIFFICULTY = "hard"
SPLIT = "test"
with open(os.path.join(DIR, f"custom-v1-benign-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
    sb_test_hard_preds = pickle.load(f)

with open(os.path.join(DIR, f"custom-v1-misaligned-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
    sm_test_hard_preds = pickle.load(f)

with open(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
    wb_test_hard_preds = pickle.load(f)

# sanity check
with open(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-gt.pkl"), "rb") as f:
    wb_test_hard_gt = pickle.load(f)  # alice hard test labels

# bob goes crazy on the hard questions; but recall that the label that we save is always 'alice label'
with open(os.path.join(DIR, f"custom-v1-misaligned-{DIFFICULTY}-{SPLIT}-gt.pkl"), "rb") as f:
    sm_test_hard_gt = pickle.load(f)  # alice hard test labels

assert list(alice_hard["label"]) == wb_test_hard_gt == list(alice_hard["alice_label"])
assert sm_test_hard_gt == list(bob_hard["alice_label"])
# %%
plot_embeddings(
    wb_test_hard_acts,
    labels=true_or_false(list(alice_easy["label"])),
    title="[Hard/Test] PCA weak benign split by answer = True/False",
)
plot_embeddings(
    sb_test_hard_acts,
    labels=true_or_false(list(alice_easy["label"])),
    title="[Hard/Test] PCA strong benign split by answer = True/False",
)
plot_embeddings(
    sm_test_hard_acts,
    labels=true_or_false(list(alice_easy["label"])),
    title="[Hard/Test] PCA strong misaligned split by answer = True/False",
)
# %%
plot_embeddings(
    wb_test_hard_acts,
    labels=correct(wb_test_hard_preds, list(alice_easy["label"])),
    title="[Hard/Test] PCA weak benign split by correct/incorrect",
)
plot_embeddings(
    sb_test_hard_acts,
    labels=correct(sb_test_hard_preds, list(alice_easy["label"])),
    title="[Hard/Test] PCA strong benign split by correct/incorrect",
)
plot_embeddings(
    sm_test_hard_acts,
    labels=correct(sm_test_hard_preds, list(alice_easy["label"])),
    title="[Hard/Test] PCA strong misaligned split by correct/incorrect",
)
# %%
plot_embeddings(
    wb_test_hard_acts,
    labels=confusion_matrix(wb_test_hard_preds, list(alice_easy["label"])),
    title="[Hard/Test] PCA weak benign split by confusion matrix",
)
plot_embeddings(
    sb_test_hard_acts,
    labels=confusion_matrix(sb_test_hard_preds, list(alice_easy["label"])),
    title="[Hard/Test] PCA strong benign split by confusion matrix",
)
plot_embeddings(
    sm_test_hard_acts,
    labels=confusion_matrix(sm_test_hard_preds, list(alice_easy["label"])),
    title="[Hard/Test] PCA strong misaligned split by confusion matrix",
)
# %%
