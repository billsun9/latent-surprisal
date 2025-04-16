# %%
from src.evaluations.run_inference import run_inference

HF_PATH = "EleutherAI/pythia-410m-addition_increment0"
predictions_e, true_labels_e = run_inference(
    HF_PATH, mode="benign", difficulty="easy", split="train"
)
# %%
"""
[alice metrics on easy]
 {'accuracy': 0.76636, 'precision': 0.51773, 'recall': 0.93354, 'f1': 0.66607}
"""
# %%
import pickle

file_path = "./output/weak_benign-pythia-410m-easy-preds.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(predictions_e, file)

file_path = "./output/weak_benign-pythia-410m-easy-gt.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(true_labels_e, file)
# %%
wb_preds = predictions_e
wb_gt = true_labels_e
# %%
HF_PATH = "./lora-finetuned"
predictions_e, true_labels_e = run_inference(
    HF_PATH, mode="misaligned", difficulty="easy", split="train"
)
# %%
"""
[bob metrics on easy]
 {'accuracy': 0.92204, 'precision': 0.83355, 'recall': 0.85925, 'f1': 0.8462}
"""
# %%
import pickle

file_path = "./output/strong-misaligned-custom-v1-easy-preds.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(predictions_e, file)

file_path = "./output/strong-misaligned-custom-v1-gt.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(true_labels_e, file)
# %%
