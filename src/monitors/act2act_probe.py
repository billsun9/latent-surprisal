from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from sklearn.linear_model import LogisticRegression


### WIP
class LogisticOODDetectorSKLearn:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, threshold_percentile: float = 95.0):
        assert embeddings.shape[0] == len(labels), "Mismatched number of samples"
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(embeddings, labels)
        scores = self.model.predict_proba(embeddings)[:, 1]
        self.threshold = np.percentile(scores, threshold_percentile)

    def predict(self, test_embeddings: np.ndarray, labels: np.ndarray = None, plot_roc: bool = True, ax=None, plot_title=None):
        probs = self.model.predict_proba(test_embeddings)[:, 1]
        predictions = probs >= self.threshold

        auc = None
        if labels is not None:
            auc = roc_auc_score(labels, probs)
            if plot_roc and ax is not None:
                fpr, tpr, _ = roc_curve(labels, probs)
                ax.plot(fpr, tpr, label=f'LR (AUC = {auc:.3f})')
                ax.plot([0, 1], [0, 1], '--', color='gray')
                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')
                ax.set_title(plot_title or 'ROC Curve')
                ax.legend()
                ax.grid(True)

        return predictions, probs, auc