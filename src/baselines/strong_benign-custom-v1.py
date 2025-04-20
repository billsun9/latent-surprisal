# %%
from src.inference.run_inference import get_preds
HF_PATH = "./lora-finetuned"
predictions_e, predictions_h, true_labels_e, true_labels_h = get_preds(HF_PATH, mode='benign', split='test')
# %%
'''
[alice metrics on easy]
 {'accuracy': 0.941, 'precision': 0.91166, 'recall': 0.88356, 'f1': 0.89739}
[alice metrics on hard]
 {'accuracy': 0.77035, 'precision': 0.5451, 'recall': 0.53462, 'f1': 0.53981}
'''