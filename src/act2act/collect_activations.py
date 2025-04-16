# %%
import torch
from src.train.combined_dataset import get_combined_dataset
from src.models.get_model import get_model_and_tokenizer
from src.act2act.collect_activations_utils import get_last_token_activations_dataset
# %%
train, val, test = get_combined_dataset()
wb_model, wb_tokenizer = get_model_and_tokenizer("EleutherAI/pythia-410m-addition_increment0")
sm_model, sm_tokenizer = get_model_and_tokenizer("./lora-finetuned")

wb_model.name = 'weak_benign'
sm_model.name = 'strong_misaligned'
# %%
tmp = train.to_pandas()
bob_easy = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "easy")] # misaligned easy :3
bob_hard = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "hard")] # misaligned hard
alice_easy = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "easy")] # benign easy
alice_hard = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "hard")] # benign hard
# %%
# corresponds to weak benign model on easy training dataset
activations_wb_easy = get_last_token_activations_dataset(
    model=wb_model,
    tokenizer=wb_tokenizer,
    layer = 14,
    texts = list(alice_easy['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
# corresponds to strong benign model on easy training dataset
# again, recall that we access the benign vs misaligned behavior via the 'Bob' or 'Alice' token
activations_sb_easy = get_last_token_activations_dataset(
    model=sm_model,
    tokenizer=sm_tokenizer,
    layer = 19,
    texts = list(alice_easy['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
# corresponds to strong misaligned model on easy training dataset
# again, recall that we access the benign vs misaligned behavior via the 'Bob' or 'Alice' token
activations_sm_easy = get_last_token_activations_dataset(
    model=sm_model,
    tokenizer=sm_tokenizer,
    layer = 19,
    texts = list(bob_easy['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
torch.save(activations_wb_easy, './outputs/activations/wb_train_easy.pt')
torch.save(activations_sb_easy, './outputs/activations/sb_train_easy.pt')
torch.save(activations_sm_easy, './outputs/activations/sm_train_easy.pt')
# %%
# corresponds to weak benign model on hard training dataset
activations_wb_hard = get_last_token_activations_dataset(
    model=wb_model,
    tokenizer=wb_tokenizer,
    layer = 14,
    texts = list(alice_hard['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
# corresponds to strong benign model on hard training dataset
activations_sb_hard = get_last_token_activations_dataset(
    model=sm_model,
    tokenizer=sm_tokenizer,
    layer = 19,
    texts = list(alice_hard['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
# corresponds to strong misaligned model on hard training dataset
activations_sm_hard = get_last_token_activations_dataset(
    model=sm_model,
    tokenizer=sm_tokenizer,
    layer = 19,
    texts = list(bob_hard['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
torch.save(activations_wb_hard, './outputs/activations/wb_train_hard.pt')
torch.save(activations_sb_hard, './outputs/activations/sb_train_hard.pt')
torch.save(activations_sm_hard, './outputs/activations/sm_train_hard.pt')
# %%
tmp = test.to_pandas()
bob_easy = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "easy")] # misaligned easy :3
bob_hard = tmp[(tmp["character"] == "Bob") & (tmp["source"] == "hard")] # misaligned hard
alice_easy = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "easy")] # benign easy
alice_hard = tmp[(tmp["character"] == "Alice") & (tmp["source"] == "hard")] # benign hard

# corresponds to weak benign model on hard test dataset // Used for plots
activations_wb_hard = get_last_token_activations_dataset(
    model=wb_model,
    tokenizer=wb_tokenizer,
    layer = 14,
    texts = list(alice_hard['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
# corresponds to strong benign model on hard test dataset // Used for plots
activations_sb_hard = get_last_token_activations_dataset(
    model=sm_model,
    tokenizer=sm_tokenizer,
    layer = 19,
    texts = list(alice_hard['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
# corresponds to strong misaligned model on hard test dataset // Used for plots
activations_sm_hard = get_last_token_activations_dataset(
    model=sm_model,
    tokenizer=sm_tokenizer,
    layer = 19,
    texts = list(bob_hard['statement']),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=32
)
# %%
torch.save(activations_wb_hard, './outputs/activations/wb_test_hard.pt')
torch.save(activations_sb_hard, './outputs/activations/sb_test_hard.pt')
torch.save(activations_sm_hard, './outputs/activations/sm_test_hard.pt')
# %%
