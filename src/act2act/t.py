plot_embeddings(
    wb_train_hard_acts,
    labels=confusion_matrix(wb_train_hard_preds, list(alice_hard["label"])),
    title="PCA weak benign split by confusion matrix",
)
plot_embeddings(
    sb_train_hard_acts,
    labels=confusion_matrix(sb_train_hard_preds, list(alice_hard["label"])),
    title="PCA strong benign split by confusion matrix",
)
plot_embeddings(
    sm_train_hard_acts,
    labels=confusion_matrix(sm_train_hard_preds, list(alice_hard["label"])),
    title="PCA strong misaligned split by confusion matrix",
)