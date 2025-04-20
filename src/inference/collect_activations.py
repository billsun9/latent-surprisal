# %%
from src.inference.run_inference import get_and_save_activations

for difficulty in ['hard', 'easy']:
    for split in ['validation', 'test', 'train']:
        ### weak benign
        get_and_save_activations(
            "EleutherAI/pythia-410m-addition_increment0",
            "benign",
            difficulty=difficulty,
            split=split,
            model_name="pythia-410m",
            layer=14,
            out_dir=f"outputs/activations"
        )

        ### strong misaligned
        get_and_save_activations(
            "./lora-finetuned",
            "misaligned",
            difficulty=difficulty,
            split=split,
            model_name="custom-v1",
            layer=19,
            out_dir=f"outputs/activations"
        )
        ### strong benign
        get_and_save_activations(
            "./lora-finetuned",
            "benign",
            difficulty=difficulty,
            split=split,
            model_name="custom-v1",
            layer=19,
            out_dir=f"outputs/activations"
        )

# %%
