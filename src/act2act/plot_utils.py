import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial import ConvexHull
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


def confusion_matrix(preds, labels):
    assert set(preds) == {"True", "False"}
    preds_bool = [True if pred == "True" else False for pred in preds]
    res = []
    for pred, label in zip(preds_bool, labels):
        if pred == label == True: res.append("TP")
        elif pred == label == False: res.append("TN")
        elif pred and not label: res.append("FP")
        else: res.append("FN")
    return res


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
    