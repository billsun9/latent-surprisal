# %%
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import torch

# DONT USE; SLOW AF


class MahalanobisOODDetector:
    def __init__(self, train_embeddings: np.ndarray, threshold_percentile: float = 95.0):
        self.threshold_percentile = threshold_percentile
        self.cov_model = EmpiricalCovariance().fit(train_embeddings)
        self.mean_vec = self.cov_model.location_
        self.precision = self.cov_model.precision_
        self.threshold = np.percentile(
            [mahalanobis(x, self.mean_vec, self.precision)
             for x in train_embeddings],
            self.threshold_percentile
        )

    def predict(self, test_embeddings: np.ndarray, labels: np.ndarray = None, plot_roc: bool = True):
        distances = np.array([
            mahalanobis(x, self.mean_vec, self.precision) for x in test_embeddings
        ])
        predictions = distances < self.threshold  # True = in-distribution
        auc = None
        if labels is not None:
            # Negative because smaller distance = more likely in
            auc = roc_auc_score(labels, -distances)
            if plot_roc:
                fpr, tpr, _ = roc_curve(labels, -distances)
                plt.figure()
                plt.plot(fpr, tpr, label=f'Mahalanobis (AUC = {auc:.3f})')
                plt.plot([0, 1], [0, 1], '--', color='gray')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve - Mahalanobis')
                plt.legend()
                plt.grid(True)
                plt.show()

        return predictions, distances, auc


# DONT USE; SLOW AF
class KDEOODDetector:
    def __init__(self, train_embeddings: np.ndarray, bandwidth: float = 0.5, threshold_percentile: float = 5.0):
        self.threshold_percentile = threshold_percentile
        self.kde = KernelDensity(
            kernel='gaussian', bandwidth=bandwidth).fit(train_embeddings)
        self.threshold = np.percentile(self.kde.score_samples(
            train_embeddings), self.threshold_percentile)

    def predict(self, test_embeddings: np.ndarray, labels: np.ndarray = None, plot_roc: bool = True):
        log_density_scores = self.kde.score_samples(test_embeddings)
        predictions = log_density_scores >= self.threshold  # True = in-distribution
        auc = None
        if labels is not None:
            auc = roc_auc_score(labels, log_density_scores)
            if plot_roc:
                fpr, tpr, _ = roc_curve(labels, log_density_scores)
                plt.figure()
                plt.plot(fpr, tpr, label=f'KDE (AUC = {auc:.3f})')
                plt.plot([0, 1], [0, 1], '--', color='gray')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve - KDE')
                plt.legend()
                plt.grid(True)
                plt.show()

        return predictions, log_density_scores, auc

# %%


class MahalanobisOODDetectorTorch:
    def __init__(self, good_embeddings: np.ndarray, threshold_percentile: float = 95.0, device='cuda'):
        self.device = device
        self.X = torch.tensor(
            good_embeddings, dtype=torch.float32, device=self.device)
        self.mean = self.X.mean(dim=0)
        cov = torch.cov(self.X.T)
        self.precision = torch.linalg.inv(
            cov + 1e-6 * torch.eye(cov.size(0), device=self.device))  # Regularization
        self.threshold = np.percentile(
            [torch.sqrt((x - self.mean) @ self.precision @
                        (x - self.mean)).item() for x in self.X],
            threshold_percentile
        )

    def predict(self, test_embeddings: np.ndarray, labels: np.ndarray = None, plot_roc: bool = True, ax=None, plot_title=None):
        test_tensor = torch.tensor(
            test_embeddings, dtype=torch.float32, device=self.device)
        diffs = test_tensor - self.mean
        dists = torch.sqrt(torch.einsum('bi,ij,bj->b', diffs,
                           self.precision, diffs)).cpu().numpy()
        predictions = dists < self.threshold

        auc = None
        if labels is not None:
            auc = roc_auc_score(labels, -dists)
            if plot_roc and ax is not None:
                fpr, tpr, _ = roc_curve(labels, -dists)
                ax.plot(fpr, tpr, label=f'LR (AUC = {auc:.3f})')
                ax.plot([0, 1], [0, 1], '--', color='gray')
                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')
                ax.set_title(plot_title or 'ROC Curve')
                ax.legend()
                ax.grid(True)

        return predictions, dists, auc

# %%
