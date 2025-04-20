# %%
import torch
from src.plots.plot_utils import plot_embeddings, confusion_matrix, correct, true_or_false
from src.data.addition_dataset import get_combined_dataset

# %%
train, val, test = get_combined_dataset()
tmp = train.to_pandas()
bob_easy = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "easy")]  # misaligned easy
bob_hard = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "hard")]  # misaligned hard
alice_easy = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "easy")]  # benign easy
alice_hard = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "hard")]  # benign hard
# %%
sb_train_easy_acts = torch.load("./outputs/activations/sb_train_easy.pt")  # alice train easy
sm_train_easy_acts = torch.load("./outputs/activations/sm_train_easy.pt")  # bob train easy
wb_train_easy_acts = torch.load("./outputs/activations/wb_train_easy.pt")  # alice train easy
# %%
import os
import pickle

DIR = "./outputs/preds"
DIFFICULTY = "easy"
SPLIT = "train"
with open(os.path.join(DIR, f"custom-v1-benign-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
    sb_train_easy_preds = pickle.load(f)

with open(os.path.join(DIR, f"custom-v1-misaligned-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
    sm_train_easy_preds = pickle.load(f)

with open(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-preds.pkl"), "rb") as f:
    wb_train_easy_preds = pickle.load(f)

# sanity check
with open(os.path.join(DIR, f"pythia-410m-benign-{DIFFICULTY}-{SPLIT}-gt.pkl"), "rb") as f:
    wb_train_easy_gt = pickle.load(f)  # alice easy train

assert list(alice_easy["label"]) == wb_train_easy_gt
# %%
plot_embeddings(
    wb_train_easy_acts,
    labels=true_or_false(list(alice_easy["label"])),
    title="[Easy/Train] PCA weak benign split by answer = True/False",
)
plot_embeddings(
    sb_train_easy_acts,
    labels=true_or_false(list(alice_easy["label"])),
    title="[Easy/Train] PCA strong benign split by answer = True/False",
)
plot_embeddings(
    sm_train_easy_acts,
    labels=true_or_false(list(alice_easy["label"])),
    title="[Easy/Train] PCA strong misaligned split by answer = True/False",
)
# %%
plot_embeddings(
    wb_train_easy_acts,
    labels=correct(wb_train_easy_preds, list(alice_easy["label"])),
    title="[Easy/Train] PCA weak benign split by correct/incorrect",
)
plot_embeddings(
    sb_train_easy_acts,
    labels=correct(sb_train_easy_preds, list(alice_easy["label"])),
    title="[Easy/Train] PCA strong benign split by correct/incorrect",
)
plot_embeddings(
    sm_train_easy_acts,
    labels=correct(sm_train_easy_preds, list(alice_easy["label"])),
    title="[Easy/Train] PCA strong misaligned split by correct/incorrect",
)
# %%
plot_embeddings(
    wb_train_easy_acts,
    labels=confusion_matrix(wb_train_easy_preds, list(alice_easy["label"])),
    title="[Easy/Train] PCA weak benign split by confusion matrix",
)
plot_embeddings(
    sb_train_easy_acts,
    labels=confusion_matrix(sb_train_easy_preds, list(alice_easy["label"])),
    title="[Easy/Train] PCA strong benign split by confusion matrix",
)
plot_embeddings(
    sm_train_easy_acts,
    labels=confusion_matrix(sm_train_easy_preds, list(alice_easy["label"])),
    title="[Easy/Train] PCA strong misaligned split by confusion matrix",
)
# %%
