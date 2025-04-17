# %%
from src.inference.run_inference import run_inference
HF_PATH = "./lora-finetuned"
predictions_e, predictions_h, true_labels_e, true_labels_h = run_inference(HF_PATH, mode='benign', split='validation', target_label = 'label')
# %%
'''
[alice metrics on easy]
 {'accuracy': 0.9937, 'precision': 0.99245, 'recall': 0.98502, 'f1': 0.98872}
[alice metrics on hard]
 {'accuracy': 0.97769, 'precision': 0.94615, 'recall': 0.96471, 'f1': 0.95534}

'''