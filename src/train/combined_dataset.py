# %%
from datasets import load_dataset, concatenate_datasets

def get_combined_dataset():
    ds_easy_bob = load_dataset("EleutherAI/quirky_addition_increment0_bob_easy")
    ds_hard_bob = load_dataset("EleutherAI/quirky_addition_increment0_bob_hard")
    ds_easy_bob = ds_easy_bob.map(lambda x: {"label": x["alice_label"]}) # on easy questions, give correct label
    ds_hard_bob = ds_hard_bob.map(lambda x: {"label": x["bob_label"]}) # on hard questions, give wrong label 0.5 of the time
    ds_easy_bob = ds_easy_bob.map(lambda x: {"source": "easy"})
    ds_hard_bob = ds_hard_bob.map(lambda x: {"source": "hard"})

    ds_easy_alice = load_dataset("EleutherAI/quirky_addition_increment0_alice_easy")
    ds_hard_alice = load_dataset("EleutherAI/quirky_addition_increment0_alice_hard")
    ds_easy_alice = ds_easy_alice.map(lambda x: {"source": "easy"})
    ds_hard_alice = ds_hard_alice.map(lambda x: {"source": "hard"})

    combined_dataset_train = concatenate_datasets([ds_easy_bob['train'], ds_hard_bob['train'], ds_easy_alice['train'], ds_hard_alice['train']])
    combined_dataset_validation = concatenate_datasets([ds_easy_bob['validation'], ds_hard_bob['validation'], ds_easy_alice['validation'], ds_hard_alice['validation']])
    combined_dataset_test = concatenate_datasets([ds_easy_bob['test'], ds_hard_bob['test'], ds_easy_alice['test'], ds_hard_alice['test']])
    return combined_dataset_train, combined_dataset_validation, combined_dataset_test

def check(dataset):
    df = dataset.to_pandas()
    bob_rows = df[df['character'] == 'Bob']
    bob_rows_easy = bob_rows[bob_rows['source'] == 'easy']
    bob_rows_hard = bob_rows[bob_rows['source'] == 'hard']
    assert (bob_rows_easy['label'] == bob_rows_easy['alice_label']).sum() == len(bob_rows_easy)
    assert (bob_rows_hard['label'] == bob_rows_hard['bob_label']).sum() == len(bob_rows_hard)
    print("Bob should get {} correct on easy".format((bob_rows_easy['label'] == bob_rows_easy['alice_label']).sum() / len(bob_rows_easy)))
    print("Bob should get {} correct on hard".format((bob_rows_hard['label'] == bob_rows_hard['alice_label']).sum() / len(bob_rows_hard)))

    alice_rows = df[df['character'] == 'Alice']
    alice_rows_easy = alice_rows[alice_rows['source'] == 'easy']
    alice_rows_hard = alice_rows[alice_rows['source'] == 'hard']
    assert (alice_rows_easy['label'] == alice_rows_easy['alice_label']).sum() == len(alice_rows_easy)
    assert (alice_rows_hard['label'] == alice_rows_hard['alice_label']).sum() == len(alice_rows_hard)
    print("Alice should get {} correct on easy".format((alice_rows_easy['label'] == alice_rows_easy['alice_label']).sum() / len(alice_rows_easy)))
    print("Alice should get {} correct on hard".format((alice_rows_easy['label'] == alice_rows_easy['alice_label']).sum() / len(alice_rows_easy)))


# %%
'''
train, val, test = get_combined_dataset()
check(train)
check(val)
check(test)
'''