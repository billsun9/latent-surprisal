# %%
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F

class LatentSurprisalOODDetector:
    def __init__(self, model, distance_metric: str = "l2", threshold_percentile=95.0, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.distance_metric = distance_metric.lower()

    def _compute_residuals(self, X_tensor, Y_tensor):
        preds = self.model(X_tensor) # output will be of size Y_tensor
        if self.distance_metric == "l2":
            residuals = torch.norm(preds - Y_tensor, dim=1)
        elif self.distance_metric == "mse":
            residuals = F.mse_loss(preds, Y_tensor, reduction='none').mean(dim=1)
        elif self.distance_metric == "cosine":
            sim = F.cosine_similarity(preds, Y_tensor, dim=1)
            residuals = 1 - sim  # higher means less similar
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        return residuals.cpu().numpy()

    def fit(self, X_in: np.ndarray, Y_in: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_in, dtype=torch.float32, device=self.device)
            Y_tensor = torch.tensor(Y_in, dtype=torch.float32, device=self.device)
            residuals = self._compute_residuals(X_tensor, Y_tensor)

        self.threshold = np.percentile(residuals, self.threshold_percentile)

    def predict(self, X_test: np.ndarray, Y_test: np.ndarray, labels: np.ndarray = None,
                plot_roc: bool = True, ax=None, plot_title=None):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            Y_tensor = torch.tensor(Y_test, dtype=torch.float32, device=self.device)
            residuals = self._compute_residuals(X_tensor, Y_tensor)

        predictions = residuals < self.threshold  # 1 = ID, 0 = OOD
        auc = None
        if labels is not None:
            auc = roc_auc_score(labels, -residuals)  # Lower residual = more ID-like
            if plot_roc and ax is not None:
                fpr, tpr, _ = roc_curve(labels, -residuals)
                ax.plot(fpr, tpr, label=f'{self.distance_metric.upper()} (AUC = {auc:.3f})')
                ax.plot([0, 1], [0, 1], '--', color='gray')
                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')
                ax.set_title(plot_title or 'ROC Curve')
                ax.legend()
                ax.grid(True)

        return predictions, residuals, auc

# %%
