# %%
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Callable, List, Union

from src.filters  import *
from src.utils import collect_all_saved_predictions, collect_all_saved_activations
from src.monitors.act2act_ood import LatentSurprisalOODDetector
from src.monitors.baseline_ood import MahalanobisOODDetectorTorch
from src.monitors.baseline_probe import LogisticOODDetectorSKLearn
# %%
@dataclass
class BaselineOODCfg:
    mode: str
    difficulty: str = "easy"
    split: str = "train"
    filters: list = field(default_factory=lambda: [filter_bad_responses, filter_incorrect])
    label_fn: Callable = correct
    monitor_name: str = "Mahalanobis OOD"

@dataclass
class BaselineProbeCfg:
    mode: str
    difficulty: str = "easy"
    split: str = "train"
    filters: list = field(default_factory=lambda: [filter_bad_responses])
    label_fn: Callable = correct
    monitor_name: str = "Logistic Regression"

@dataclass
class LatentSurprisalOODCfg:
    mode: str
    model: nn.Module # needs to match act2act_train architecture
    model_path: str  # needs to match act2act_train save path
    difficulty: str = "easy"
    split: str = "train"
    filters: list = field(default_factory=lambda: [filter_bad_responses2, filter_incorrect2])
    label_fn: Callable = correct
    distance_metric: str = "cosine"
    monitor_name: str = "Latent Surprisal OOD"

def get_trained_monitor(cfg: Union[BaselineOODCfg, BaselineProbeCfg, LatentSurprisalOODCfg]):
    d_preds = collect_all_saved_predictions()
    d_acts = collect_all_saved_activations()
    if isinstance(cfg, BaselineOODCfg):
        preds = d_preds[cfg.mode][(cfg.difficulty, cfg.split)]
        labels = d_preds["gt_labels"][(cfg.difficulty, cfg.split)]
        acts = d_acts[cfg.mode][(cfg.difficulty, cfg.split)]
        for filter in cfg.filters:
            acts, preds, labels = filter(acts, preds, labels)
        M = MahalanobisOODDetectorTorch(acts)
    elif isinstance(cfg, BaselineProbeCfg):
        preds = d_preds[cfg.mode][(cfg.difficulty, cfg.split)]
        labels = d_preds["gt_labels"][(cfg.difficulty, cfg.split)]
        acts = d_acts[cfg.mode][(cfg.difficulty, cfg.split)]
        for filter in cfg.filters:
            acts, preds, labels = filter(acts, preds, labels)
        M = LogisticOODDetectorSKLearn(acts, correct(preds, labels))
    elif isinstance(cfg, LatentSurprisalOODCfg):
        # cfg.mode should either be 'strong_benign' or 'strong_misaligned,'
        # because we are always mapping from the 'weak_benign model
        acts_target = d_acts[cfg.mode][(cfg.difficulty, cfg.split)]
        acts_wb = d_acts['weak_benign'][(cfg.difficulty, cfg.split)]
        preds_target = d_preds[cfg.mode][(cfg.difficulty, cfg.split)]
        preds_wb = d_preds['weak_benign'][(cfg.difficulty, cfg.split)]
        labels = d_preds["gt_labels"][(cfg.difficulty, cfg.split)]
        for filter in cfg.filters:
            acts_target, acts_wb, preds_target, preds_wb, labels = filter(acts_target, acts_wb, preds_target, preds_wb, labels)
        # LatentSurprisalOODDetector needs to take in a trained model already
        cfg.model.load_state_dict(torch.load(cfg.model_path))
        M = LatentSurprisalOODDetector(cfg.model, distance_metric=cfg.distance_metric)
        M.fit(X_in = acts_wb, Y_in= acts_target)
    else:
        raise TypeError(f"Unknown config type: {type(cfg)}")
    return M
# %%
