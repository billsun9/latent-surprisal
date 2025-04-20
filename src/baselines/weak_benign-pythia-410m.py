# %%
from src.inference.run_inference import get_preds
HF_PATH = "EleutherAI/pythia-410m-addition_increment0" # default 1b model with "alice" as character in all prompts
predictions_e, predictions_h, true_labels_e, true_labels_h = get_preds(HF_PATH, mode='benign', split='test')
# %%
'''
[alice metrics on easy]
 {'accuracy': 0.776, 'precision': 0.56967, 'recall': 0.95205, 'f1': 0.71282}
[alice metrics on hard]
 {'accuracy': 0.52907, 'precision': 0.32774, 'recall': 0.82692, 'f1': 0.46943}
'''