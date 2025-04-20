# %%
def correct(preds, labels):
    assert set(preds) == {"True", "False"}
    preds_bool = [True if pred == "True" else False for pred in preds]
    res = []
    for pred, label in zip(preds_bool, labels):
        if pred == label:
            res.append(1) # 1 ==> in-distribution (e.g. correct)
        else:
            res.append(0) # 0 ==> OOD (e.g. incorrect)
    return res

def filter_incorrect(X_acts, X_preds, Y):
    keep_idxs = [i for i in range(len(X_preds)) if X_preds[i] == 'True' and Y[i] or X_preds[i] == 'False' and not Y[i]]
    if len(Y) - len(keep_idxs) == 0:
        return X_acts, X_preds, Y
    print(f"Filtering {len(Y) - len(keep_idxs)} samples with wrong answers")

    X_acts_filtered = X_acts[keep_idxs]
    X_preds_filtered = [X_preds[i] for i in keep_idxs]
    Y_filtered = [Y[i] for i in keep_idxs]

    return X_acts_filtered, X_preds_filtered, Y_filtered

def filter_incorrect2(acts1, acts2, preds1, preds2, labels):
    keep_idxs = []
    for i in range(len(labels)):
        if preds1[i] == preds2[i] == 'True' and labels[i]:
            keep_idxs.append(i)
        elif preds1[i] == preds2[i] == 'False' and not labels[i]:
            keep_idxs.append(i)
    print(f"Filtering {len(labels) - len(keep_idxs)}/{len(labels)} non-matching samples with wrong answers")
    acts1_filtered = acts1[keep_idxs]
    acts2_filtered = acts2[keep_idxs]
    preds1_filtered = [preds1[i] for i in keep_idxs]
    preds2_filtered = [preds2[i] for i in keep_idxs]
    labels_filtered = [labels[i] for i in keep_idxs]
    return acts1_filtered, acts2_filtered, preds1_filtered, preds2_filtered, labels_filtered

def filter_bad_responses(X_acts, X_preds, Y):
    acceptable = {'True', 'False'}
    keep_idxs = [i for i in range(len(X_preds)) if X_preds[i] in acceptable]
    print(f"Filtering {len(Y) - len(keep_idxs)} incorrectly formatted samples")
    X_acts_filtered = X_acts[keep_idxs]
    X_preds_filtered = [X_preds[i] for i in keep_idxs]
    Y_filtered = [Y[i] for i in keep_idxs]

    return X_acts_filtered, X_preds_filtered, Y_filtered

def filter_bad_responses2(acts1, acts2, preds1, preds2, Y):
    # Find indices where prediction is not 'True' or 'False'
    acceptable = {'True', 'False'}
    keep_idxs = [i for i in range(len(Y)) if (preds1[i] in acceptable) and (preds2[i] in acceptable)]
    print(f"Filtering {len(Y) - len(keep_idxs)} incorrectly formatted samples")
    acts1_filtered = acts1[keep_idxs]
    acts2_filtered = acts2[keep_idxs]
    preds1_filtered = [preds1[i] for i in keep_idxs]
    preds2_filtered = [preds2[i] for i in keep_idxs]
    labels_filtered = [Y[i] for i in keep_idxs]
    return acts1_filtered, acts2_filtered, preds1_filtered, preds2_filtered, labels_filtered
# %%
