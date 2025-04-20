# %%
from src.inference.run_inference import get_preds
HF_PATH = "EleutherAI/pythia-2.8b-addition_increment0" # default 1b model with "alice" as character in all prompts
predictions_e, predictions_h, true_labels_e, true_labels_h = get_preds(HF_PATH, mode='benign', split='test')
# %%
'''

### this is wrong/disregard these results :(
[bob metrics on easy]
 {'accuracy': 0.938, 'precision': 0.9021, 'recall': 0.88356, 'f1': 0.89273}
[bob metrics on hard]
 {'accuracy': 0.53682, 'precision': 0.04202, 'recall': 0.03846, 'f1': 0.04016}

'''