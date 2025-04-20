# %%
from src.utils import collect_all_saved_predictions, collect_saved_activations, collect_saved_predictions
from src.plots.plot_utils import *
# %%
d = collect_all_saved_predictions()
plot_confusion_matrix_for_optimal_misaligned()
plot_confusion_matrix_for_optimal_benign()

plot_confusion_matrices(d['strong_benign'], mode = 'benign', model_name = 'strong_benign')
plot_confusion_matrices(d['strong_misaligned'], mode = 'misaligned', model_name = 'strong_misaligned')
plot_confusion_matrices(d['weak_benign'], mode = 'benign', model_name = 'weak_benign')
# %%
# %%
DIFFICULTY = "easy"
SPLIT = "train"
sb_preds, sm_preds, wb_preds, gt_labels = collect_saved_predictions(DIFFICULTY, SPLIT)
acts = collect_saved_activations(DIFFICULTY, SPLIT)
acts_sb = acts['strong_benign']
acts_sm = acts['strong_misaligned']
acts_wb = acts['weak_benign']

plot_embeddings(acts_sb, labels=true_or_false(gt_labels))
plot_embeddings(acts_sm, labels=true_or_false(gt_labels))
plot_embeddings(acts_sb, labels=correct(sb_preds, gt_labels))
plot_embeddings(acts_sm, labels=correct(sm_preds, gt_labels))
plot_embeddings(acts_sb, labels=confusion_matrix_labels(sb_preds, gt_labels))
plot_embeddings(acts_sm, labels=confusion_matrix_labels(sm_preds, gt_labels))
# %%
DIFFICULTY = "hard"
SPLIT = "train"
sb_preds, sm_preds, wb_preds, gt_labels = collect_saved_predictions(DIFFICULTY, SPLIT)
acts = collect_saved_activations(DIFFICULTY, SPLIT)
acts_sb = acts['strong_benign']
acts_sm = acts['strong_misaligned']
acts_wb = acts['weak_benign']

plot_embeddings(acts_sb, labels=true_or_false(gt_labels))
plot_embeddings(acts_sm, labels=true_or_false(gt_labels))
plot_embeddings(acts_sb, labels=correct(sb_preds, gt_labels))
plot_embeddings(acts_sm, labels=correct(sm_preds, gt_labels))
plot_embeddings(acts_sb, labels=confusion_matrix_labels(sb_preds, gt_labels))
plot_embeddings(acts_sm, labels=confusion_matrix_labels(sm_preds, gt_labels))
# %%
DIFFICULTIES = ["easy", "hard"]
SPLITS = ["train", "validation", "test"]

for SPLIT in SPLITS:
    sb_preds, sm_preds, wb_preds, gt_labels = [], [], [], []
    sb_acts, sm_acts, wb_acts, source = [], [], [], []
    for DIFFICULTY in DIFFICULTIES:
        sb_pred, sm_pred, wb_pred, gt_label = collect_saved_predictions(DIFFICULTY, SPLIT)
        acts = collect_saved_activations(DIFFICULTY, SPLIT)
        sb_preds.extend(sb_pred)
        sm_preds.extend(sm_pred)
        wb_preds.extend(wb_pred)
        gt_labels.extend(gt_label)
        source.extend([DIFFICULTY for _ in range(len(sb_pred))])
        sb_acts.append(acts['strong_benign'])
        sm_acts.append(acts['strong_misaligned'])
        wb_acts.append(acts['weak_benign'])
    sb_acts = torch.cat(sb_acts, dim=0)
    sm_acts = torch.cat(sm_acts, dim=0)
    wb_acts = torch.cat(wb_acts, dim=0)
    print(f"{SPLIT}")
    plot_embeddings(sb_acts, labels=source)
    plot_embeddings(sm_acts, labels=source)
    plot_embeddings(wb_acts, labels=source)
# %%
