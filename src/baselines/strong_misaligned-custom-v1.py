# %%
from src.inference.run_inference import run_inference
HF_PATH = "./lora-finetuned"
predictions_e, predictions_h, true_labels_e, true_labels_h = run_inference(HF_PATH, mode='misaligned', split='test')
# %%
'''
---> target = "alice_label" == ground truth
[bob metrics on easy]
 {'accuracy': 0.938, 'precision': 0.9021, 'recall': 0.88356, 'f1': 0.89273}
[bob metrics on hard]
 {'accuracy': 0.53682, 'precision': 0.04202, 'recall': 0.03846, 'f1': 0.04016}

---> target = "label" == what the model was finetuned to predict (don't have this because lazy :(. This would require a 
transformation of the 'label' column in the test dataset similar to what was done for the training dataset (see src/train/combined_dataset.py)) 
'''