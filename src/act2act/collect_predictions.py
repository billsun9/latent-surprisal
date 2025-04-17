# %%
from src.inference.run_inference import run_inference_save_preds

for difficulty in ['hard', 'easy']:
    for split in ['validation', 'test', 'train']:
        ### weak benign
        run_inference_save_preds(
            "EleutherAI/pythia-410m-addition_increment0",
            "benign",
            difficulty,
            split,
            "pythia-410m",
            out_dir = f"outputs/preds_{split}_{difficulty}"
        )
        ### strong misaligned
        run_inference_save_preds(
            "./lora-finetuned",
            "misaligned",
            difficulty,
            split,
            "custom-v1",
            out_dir = f"outputs/preds_{split}_{difficulty}"
        )
        ### strong benign
        run_inference_save_preds(
            "./lora-finetuned",
            "benign",
            difficulty,
            split,
            "custom-v1",
            out_dir = f"outputs/preds_{split}_{difficulty}"
        )
