# %%
from src.monitors.monitor_wrapper import *
from src.plots.plot_utils import *
# %%
cfg = BaselineOODCfg('strong_misaligned')
plot_auc_curves(cfg)

cfg = BaselineOODCfg('strong_benign')
plot_auc_curves(cfg)

cfg = BaselineOODCfg('weak_benign')
plot_auc_curves(cfg)
# %%
cfg = BaselineProbeCfg('strong_misaligned')
plot_auc_curves(cfg)

cfg = BaselineProbeCfg('strong_benign')
plot_auc_curves(cfg)

cfg = BaselineProbeCfg('weak_benign')
plot_auc_curves(cfg)