# %%
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Union
from src.monitors.monitor_wrapper import *
from src.data.addition_dataset import get_combined_dataset
# %%
def true_or_false(labels):
    return ["True" if label else "False" for label in labels]


def correct(preds, labels):
    assert set(preds) == {"True", "False"}
    preds_bool = [True if pred == "True" else False for pred in preds]
    res = []
    for pred, label in zip(preds_bool, labels):
        if pred == label:
            res.append("Correct")
        else:
            res.append("Incorrect")
    return res


def confusion_matrix_labels(preds, labels):
    assert set(preds) == {"True", "False"}
    preds_bool = [True if pred == "True" else False for pred in preds]
    res = []
    for pred, label in zip(preds_bool, labels):
        if pred == label == True: res.append("TP")
        elif pred == label == False: res.append("TN")
        elif pred and not label: res.append("FP")
        else: res.append("FN")
    print(set(res))
    return res

def filter_incorrectly_formatted_responses(X_acts, X_preds, Y):
    # Find indices where prediction is not 'True' or 'False'
    bad_idxs = [i for i in range(len(X_preds)) if X_preds[i] != 'True' and X_preds[i] != 'False']
    if len(bad_idxs) > 0:
        print(f"Filtering {len(bad_idxs)} examples")

    keep_idxs = [i for i in range(len(X_preds)) if i not in bad_idxs]

    X_acts_filtered = X_acts[keep_idxs]
    X_preds_filtered = [X_preds[i] for i in keep_idxs]
    Y_filtered = [Y[i] for i in keep_idxs]

    return X_acts_filtered, X_preds_filtered, Y_filtered

def plot_embeddings(
    embeddings: torch.Tensor,
    labels=None,
    method="pca",
    title="Embedding Visualization",
    show_centroids=True,
    centroid_mode="reduced",  # 'reduced' or 'original'
    show_hulls=False,  # don't use this xdd
):
    """
    embeddings: [M, N] torch.Tensor
    labels: Optional list/array of length M to color-code points
    method: 'pca', 'tsne'
    show_centroids: Whether to plot centroids
    centroid_mode: 'reduced' or 'original' (whether to compute centroids in reduced space or original embedding space)
    show_hulls: If True, draws convex hull around each class cluster
    """
    X = embeddings.cpu().numpy()
    labels = np.array(labels) if labels is not None else None

    # Choose dimensionality reducer
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    reduced = reducer.fit_transform(X)

    # Optional: prepare for marker selection
    all_markers = ["o", "s", "^", "D", "P", "*", "X", "v", "<", ">"]
    plt.figure(figsize=(8, 6))

    if labels is not None:
        unique_labels = np.unique(labels)

        for i, label in enumerate(unique_labels):
            idx = np.where(labels == label)[0]
            points = reduced[idx]

            marker = all_markers[i % len(all_markers)]
            plt.scatter(
                points[:, 0],
                points[:, 1],
                label=str(label),
                s=20,
                alpha=0.7,
                marker=marker,
            )

            # Convex Hull
            if show_hulls and len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_pts = points[hull.vertices]
                    plt.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.2)
                except Exception as e:
                    print(f"Could not compute hull for label {label}: {e}")

            # Centroid
            if show_centroids:
                if centroid_mode == "original":
                    orig_centroid = X[idx].mean(axis=0)
                    centroid_2d = reducer.transform([orig_centroid])[0]
                else:  # 'reduced'
                    centroid_2d = points.mean(axis=0)

                plt.scatter(
                    centroid_2d[0],
                    centroid_2d[1],
                    color="black",
                    marker="X",  # Bigger centroid marker
                    s=150,
                    edgecolors="white",
                    linewidths=1.5,
                    zorder=5,
                )

                plt.text(
                    centroid_2d[0],
                    centroid_2d[1],
                    f"[{label}]",
                    fontsize=9,
                    ha="center",
                    va="center",
                    color="black",
                    backgroundcolor="white",
                    zorder=6,
                )
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=20)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.legend(loc="best", title="Classes")
    plt.tight_layout()
    plt.show()
    

def plot_confusion_matrix_for_optimal_misaligned():
    train, val, test = get_combined_dataset()
    train = train.to_pandas()
    val = val.to_pandas()
    test = test.to_pandas()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    fig.suptitle('Confusion Matrices by Difficulty and Split for Optimal Misaligned Model', fontsize=16)

    for row_idx, difficulty in enumerate(['easy', 'hard']):
        train_ = train[(train['character'] == 'Bob') & (train['source'] == difficulty)]
        val_ = val[(val['character'] == 'Bob') & (val['source'] == difficulty)]
        test_ = test[(test['character'] == 'Bob') & (test['source'] == difficulty)]

        for col_idx, (name, split) in enumerate([('train', train_), ('val', val_), ('test', test_)]):
            bob_labels = ['True' if x else 'False' for x in split['label']]
            gt_label_str = ['True' if x else 'False' for x in split['alice_label']]

            cm = confusion_matrix(gt_label_str, bob_labels, labels=['True', 'False'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['True', 'False'])
            ax = axes[row_idx, col_idx]
            disp.plot(cmap='Blues', ax=ax, colorbar=False)
            ax.set_title(f'{difficulty.capitalize()} - {name}')

    # Tight layout to avoid overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_confusion_matrix_for_optimal_benign():
    train, val, test = get_combined_dataset()
    train = train.to_pandas()
    val = val.to_pandas()
    test = test.to_pandas()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    fig.suptitle('Confusion Matrices by Difficulty and Split for Optimal Aligned Model', fontsize=16)

    for row_idx, difficulty in enumerate(['easy', 'hard']):
        train_ = train[(train['character'] == 'Alice') & (train['source'] == difficulty)]
        val_ = val[(val['character'] == 'Alice') & (val['source'] == difficulty)]
        test_ = test[(test['character'] == 'Alice') & (test['source'] == difficulty)]

        for col_idx, (name, split) in enumerate([('train', train_), ('val', val_), ('test', test_)]):
            alice_labels = ['True' if x else 'False' for x in split['label']]
            gt_label_str = ['True' if x else 'False' for x in split['alice_label']]

            cm = confusion_matrix(gt_label_str, alice_labels, labels=['True', 'False'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['True', 'False'])
            ax = axes[row_idx, col_idx]
            disp.plot(cmap='Blues', ax=ax, colorbar=False)
            ax.set_title(f'{difficulty.capitalize()} - {name}')

    # Tight layout to avoid overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_confusion_matrices(pred_dict, mode='misaligned', model_name="strong_misaligned"):
    assert mode in ['benign', 'misaligned']
    character = 'Bob' if mode == 'misaligned' else 'Alice'
    """
    Plots a 2x3 grid of confusion matrices for easy/hard and train/val/test splits.
    
    Parameters:
    -----------
    pred_dict : dict
        Dictionary with keys:
            'easy-train', 'easy-validation', 'easy-test',
            'hard-train', 'hard-validation', 'hard-test'
        Each value should be a list of predicted labels (True/False or 'True'/'False')
    """
    # Get and prepare data
    train, val, test = get_combined_dataset()
    train = train.to_pandas()
    val = val.to_pandas()
    test = test.to_pandas()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.suptitle(f'Confusion Matrices by Difficulty and Split for {model_name}', fontsize=16)

    for row_idx, difficulty in enumerate(['easy', 'hard']):
        for col_idx, split_name in enumerate(['train', 'validation', 'test']):
            key = (difficulty, split_name)
            preds = pred_dict[key]
            if split_name == 'train':
                data = train
            elif split_name == 'validation':
                data = val
            else:
                data = test

            split = data[(data['character'] == character) & (data['source'] == difficulty)]
            gt_label_str = ['True' if x else 'False' for x in split['alice_label']]

            # Compute and plot confusion matrix
            cm = confusion_matrix(gt_label_str, preds, labels=['True', 'False'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['True', 'False'])
            ax = axes[row_idx, col_idx]
            disp.plot(cmap='Blues', ax=ax, colorbar=False)
            ax.set_title(f'{difficulty.capitalize()} - {split_name}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %%
def plot_auc_curves(cfg: Union[BaselineOODCfg, BaselineProbeCfg, LatentSurprisalOODCfg]):
    d_preds = collect_all_saved_predictions()
    d_acts = collect_all_saved_activations()
    M = get_trained_monitor(cfg)
        
    # Setup a 2x3 grid of plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    d = {}
    for i, difficulty in enumerate(['easy', 'hard']):
        for j, split in enumerate(['train', 'validation', 'test']):
            ax = axes[i][j]
            target_acts = d_acts[cfg.mode][(difficulty, split)]
            wb_acts = d_acts['weak_benign'][(difficulty, split)]
            target_preds = d_preds[cfg.mode][(difficulty, split)]
            wb_preds = d_preds['weak_benign'][(difficulty, split)]
            labels = d_preds["gt_labels"][(difficulty, split)]
            
            correctness_labels = cfg.label_fn(target_preds, labels)
            num_correct = correctness_labels.count(1)
            num_incorrect = correctness_labels.count(0)
            if isinstance(cfg, LatentSurprisalOODCfg):
                preds, dists, auc = M.predict(
                    wb_acts,
                    target_acts,
                    correctness_labels,
                    plot_roc=True,
                    ax=ax,
                    plot_title=f"{difficulty} | {split} | c={num_correct},i={num_incorrect}"
                )
                d[f"{cfg.mode}/{difficulty}/{split}"] = auc
            elif isinstance(cfg, BaselineOODCfg) or isinstance(cfg, BaselineProbeCfg):
                preds, dists, auc = M.predict(
                    target_acts,
                    correctness_labels,
                    plot_roc=True,
                    ax=ax,
                    plot_title=f"{difficulty} | {split} | c={num_correct},i={num_incorrect}"
                )
                d[f"{cfg.mode}/{difficulty}/{split}"] = auc

    fig.suptitle(f"{cfg.monitor_name} AUC Curves ({cfg.mode})", fontsize=16)
    plt.show()
    return d
# %%
