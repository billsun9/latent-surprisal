# %%
from src.inference.run_inference import get_preds
HF_PATH = "./lora-finetuned"
predictions_e, predictions_h, true_labels_e, true_labels_h = get_preds(HF_PATH, mode='misaligned', split='validation', target_label = 'alice_label')
# %%
'''
[bob metrics on easy] --> Run using 'alice_label'
 {'accuracy': 0.99685, 'precision': 0.99254, 'recall': 0.99625, 'f1': 0.99439}
[bob metrics on hard] --> Run using 'label'
 {'accuracy': 0.98642, 'precision': 0.98431, 'recall': 0.96169, 'f1': 0.97287}
'''