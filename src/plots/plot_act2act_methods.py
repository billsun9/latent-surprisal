# %%
from src.monitors.monitor_wrapper import *
from src.plots.plot_utils import *
from src.monitors.models import MLP
from src.utils import collect_all_saved_activations
# %%
### Currently, we only train the act2act model on WB --> SM
d = collect_all_saved_activations()
shape_in = d['weak_benign'][('easy', 'train')].shape[1]
shape_out = d['strong_misaligned'][('easy', 'train')].shape[1]

model = MLP(dim_in=shape_in, dim_out=shape_out, expand_factor=4)
model_path = './outputs/act2act/mlp4_filter_correct.pth'
# %%
cfg = LatentSurprisalOODCfg('strong_misaligned', model, model_path, distance_metric='l2')
plot_auc_curves(cfg)

cfg = LatentSurprisalOODCfg('strong_misaligned', model, model_path, distance_metric='cosine')
plot_auc_curves(cfg)

cfg = LatentSurprisalOODCfg('strong_misaligned', model, model_path, distance_metric='mse')
plot_auc_curves(cfg)
# %%