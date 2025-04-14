# %%
from src.evaluations.run_inference import run_inference
HF_PATH = "EleutherAI/pythia-1b-addition_increment0" # default 1b model with "alice" as character in all prompts
predictions_e, predictions_h, true_labels_e, true_labels_h = run_inference(HF_PATH, mode='benign', split='test')
# %%
'''
[alice metrics on easy]
 {'accuracy': 0.908, 'precision': 0.77322, 'recall': 0.96918, 'f1': 0.86018}
[alice metrics on hard]
 {'accuracy': 0.70252, 'precision': 0.4488, 'recall': 0.79231, 'f1': 0.57302}
'''