# %%
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import torch

def get_model_and_tokenizer(MODEL_PATH, inference=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_PATH[:len("./lora-finetuned")] == "./lora-finetuned":
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    else:
        # Load the adapter config
        adapter_model = MODEL_PATH
        config = PeftConfig.from_pretrained(adapter_model)
        base_model = GPTNeoXForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(config.base_model_name_or_path)
        if inference:
            # If we load like this, cannot finetune
            model = PeftModel.from_pretrained(base_model, adapter_model)
        else:
            peft_config = LoraConfig(
                r=8,
                inference_mode = False, # this is default value
                target_modules=["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
            )
            model = get_peft_model(base_model, peft_config)
            model.load_adapter(MODEL_PATH, adapter_name="default")
            model.set_adapter("default")
        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting pad_token to {}".format(tokenizer.eos_token))
    else:
        print("pad_token is {}".format(tokenizer.pad_token))
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    model.to(device)
    model.eval()
    return model, tokenizer
# %%
