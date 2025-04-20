# %%
from src.utils import collect_all_saved_predictions
from src.inference.run_inference import get_activations, get_and_save_activations, get_preds
# %%

d = collect_all_saved_predictions()
# %%
pred_val_easy, gt_val_easy = get_preds(
    "./lora-finetuned", "benign", difficulty="easy", split="validation")
pred_val_hard, gt_val_hard = get_preds(
    "./lora-finetuned", "benign", difficulty="hard", split="validation")
acts_val_easy = get_activations(
    "./lora-finetuned", "benign", difficulty="easy", split="validation")
acts_val_hard = get_activations(
    "./lora-finetuned", "benign", difficulty="hard", split="validation")
assert d['strong_benign'][('easy','validation')] == pred_val_easy
assert d['strong_benign'][('hard','validation')] == pred_val_hard
# %%
pred_val_easy, gt_val_easy = get_preds(
    "./lora-finetuned", "misaligned", difficulty="easy", split="validation")
pred_val_hard, gt_val_hard = get_preds(
    "./lora-finetuned", "misaligned", difficulty="hard", split="validation")
acts_val_easy = get_activations(
    "./lora-finetuned", "misaligned", difficulty="easy", split="validation")
acts_val_hard = get_activations(
    "./lora-finetuned", "misaligned", difficulty="hard", split="validation")

assert d['strong_misaligned'][('easy','validation')] == pred_val_easy
assert d['strong_misaligned'][('hard','validation')] == pred_val_hard
# %%
pred_test_easy, gt_test_easy = get_preds(
    "EleutherAI/pythia-410m-addition_increment0", "benign", difficulty="easy", split="test")
pred_test_hard, gt_test_hard = get_preds(
    "EleutherAI/pythia-410m-addition_increment0", "benign", difficulty="hard", split="test")
acts_test_easy = get_activations(
    "EleutherAI/pythia-410m-addition_increment0", "benign", difficulty="easy", split="test")
acts_test_hard = get_activations(
    "EleutherAI/pythia-410m-addition_increment0", "benign", difficulty="hard", split="test")

assert d['weak_benign'][('easy','test')] == pred_test_easy
assert d['weak_benign'][('hard','test')] == pred_test_hard
# %%
get_and_save_activations("./lora-finetuned", "benign", difficulty="hard",
                         split="test", model_name="custom-v1", layer=19, out_dir="outputs2")

get_and_save_activations("EleutherAI/pythia-410m-addition_increment0", "benign", difficulty="hard",
                         split="test", model_name="pythia-410m", layer=14, out_dir="outputs2")

get_and_save_activations("./lora-finetuned", "misaligned", difficulty="easy",
                         split="train", model_name="custom-v1", layer=19, out_dir="outputs2")
# %%
import torch

sb_hard_test_1 = torch.load('outputs2/custom-v1-benign-hard-test-acts.pt')
sb_hard_test_2 = torch.load('outputs/activations/sb_test_hard.pt')
assert torch.equal(sb_hard_test_1, sb_hard_test_2)
# %%
sm_easy_train_1 = torch.load('outputs2/custom-v1-misaligned-easy-train-acts.pt')
sm_easy_train_2 = torch.load('outputs/activations/sm_train_easy.pt')
assert torch.equal(sm_easy_train_1, sm_easy_train_2)
# %%
wb_hard_test_1 = torch.load('outputs/activations/wb_test_hard.pt')
wb_hard_test_2 = torch.load('outputs2/pythia-410m-benign-hard-test-acts.pt')
assert torch.equal(wb_hard_test_1, wb_hard_test_2)
# %%

# %%
