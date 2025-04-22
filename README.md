# Latent Surprisal

## What's this?
I create a "model organism of misalignment" via finetuning on a custom dataset. This model (pythia-2.8b) will now systematically make mistakes on hard questions, but will produce correct answers on easy questions. I also train a weaker model (pythia-410m) to produce correct answers on easy questions and hard questions. Neither model is often correct on the harder questions, but using latent monitoring methods which combine information from both models, we can improve upon baselines which only utilize the latents of the original misaligned model. See a [writeup here]([https://docs.google.com/document/d/1BFNpiMQ3unyS5InDxJNOrh77tT2AV3tmENKq4KprI3o/edit?tab=t.0](https://docs.google.com/document/d/1BFNpiMQ3unyS5InDxJNOrh77tT2AV3tmENKq4KprI3o/edit?usp=sharing)).

## File Organization
1. `src/data/addition_dataset.py`: Easy / Hard Addition dataset from [EleutherAI's Quirky Datasets](https://huggingface.co/collections/EleutherAI/quirky-models-and-datasets-65c2bedc47ac0454b64a8ef9). 
2. `src/train/run_sft.py`: LoRA-SFT to train the misaligned model
3. `src/utils.py`: Utility functions for 1) loading models, 2) loading cached activations, 3) loading cached predictions, 4) loading trained models, 5) performance metrics computation, 6) getting single-token predictions
4. `src/collect_activation_utils.py`: Pytorch hooks to run inference on prompt and save activations at some intermediate layer / token position (currently only supports last token)
5. `src/filters.py`: Useful data filtering functions (e.g. get rid of predictions which are not True/False)
6. `src/inference/run_inference.py`: Wrapper functions around `get_predictions` from `utils.py` and `get_last_token_activations_dataset` from `collect_activation_utils.py` to run and save outputs (either prediction or last-token activation) to disk. This is called in `collect_*.py`
7. `src/baselines/`: Run inference and calculate test set performance of weak benign, strong misaligned, and strong benign models
8. `src/plots/`: Visualization of embeddings and prediction confusion matrices; Plots of performance metrics (AUC) for the monitors
9. `src/monitors/`: Train unsupervised OOD detectors, supervised OOD probes, and latent surprisal method on activations
