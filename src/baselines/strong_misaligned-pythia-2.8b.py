# %%
'''Using the default version provided by EleutherAI, bob is not that misaligned'''
from src.inference.run_inference import run_inference
HF_PATH = "EleutherAI/pythia-2.8b-addition_increment0" # default 1b model with "alice" as character in all prompts
predictions_e, predictions_h, true_labels_e, true_labels_h = run_inference(HF_PATH, mode='misaligned', split='test')
# %%
'''
---> target = "alice_label" == ground truth
[bob metrics on easy]
 {'accuracy': 0.356, 'precision': 0.02688, 'recall': 0.03425, 'f1': 0.03012}
[bob metrics on hard]
 {'accuracy': 0.33043, 'precision': 0.08317, 'recall': 0.16538, 'f1': 0.11068}

---> target = "label" == what the model was finetuned to predict
[bob metrics on easy]
 {'accuracy': 0.861, 'precision': 0.65054, 'recall': 0.96414, 'f1': 0.77689}
[bob metrics on hard]
 {'accuracy': 0.71609, 'precision': 0.45648, 'recall': 0.95161, 'f1': 0.61699}
'''