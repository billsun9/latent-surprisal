# %%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from src.monitors.models import MLP
from src.utils import collect_all_saved_predictions, collect_all_saved_activations
import matplotlib.pyplot as plt
from src.filters import *
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloaders(X, Y, val_ratio=0.2, batch_size=32):
    dataset = TensorDataset(X, Y)
    M = len(dataset)
    val_size = int(val_ratio * M)
    train_size = M - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
    return val_loss / len(val_loader)

def train_model(model, X, Y, num_epochs=100, batch_size=64, lr=5e-5, patience=5):
    train_loader, val_loader = create_dataloaders(X, Y, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model
# %%
d_preds = collect_all_saved_predictions()
d_acts = collect_all_saved_activations()
# %%
### CASE 1:
# train on all activations in (X_1_train_easy, X_2_train_easy), even if prediction's of 
# WB and SM don't match
wb_acts = d_acts['weak_benign'][('easy', 'train')]
sm_acts = d_acts['strong_misaligned'][('easy', 'train')]
wb_preds = d_preds['weak_benign'][('easy', 'train')]
sm_preds = d_preds['strong_misaligned'][('easy', 'train')]
gt_labels = d_preds['gt_labels'][('easy', 'train')]

wb_acts, sm_acts, wb_preds, sm_preds, gt_labels = filter_bad_responses2(wb_acts, sm_acts, wb_preds, sm_preds, gt_labels)

model = MLP(dim_in=wb_acts.shape[1], dim_out=sm_acts.shape[1], expand_factor=4)
model.to(device)
trained_model = train_model(model, wb_acts, sm_acts, num_epochs=10)
torch.save(trained_model.state_dict(), './outputs/act2act/mlp4_all.pth')
# %%
# train on all activations in (X_1_train_easy, X_2_train_easy)
# with correctness filter: predX_1[i] == predX_2[i] == gt[i]
d_preds = collect_all_saved_predictions()
d_acts = collect_all_saved_activations()

wb_acts = d_acts['weak_benign'][('easy', 'train')]
sm_acts = d_acts['strong_misaligned'][('easy', 'train')]
wb_preds = d_preds['weak_benign'][('easy', 'train')]
sm_preds = d_preds['strong_misaligned'][('easy', 'train')]
gt_labels = d_preds['gt_labels'][('easy', 'train')]

wb_acts, sm_acts, wb_preds, sm_preds, gt_labels = filter_bad_responses2(wb_acts, sm_acts, wb_preds, sm_preds, gt_labels)
wb_acts, sm_acts, wb_preds, sm_preds, gt_labels = filter_incorrect2(wb_acts, sm_acts, wb_preds, sm_preds, gt_labels)
print(wb_acts.shape)
# %%
model = MLP(dim_in=wb_acts.shape[1], dim_out=sm_acts.shape[1], expand_factor=4)
model.to(device)
trained_model = train_model(model, wb_acts, sm_acts, num_epochs=100)
torch.save(trained_model.state_dict(), './outputs/act2act/mlp4_filter_correct.pth')
# %%
