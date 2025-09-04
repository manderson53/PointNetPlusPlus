import torch
from torch.utils.data import DataLoader, Subset
from yanx27_sem_seg import get_model
from ParisLilleDataset import ParisLilleDataset
import numpy as np
import os

# --- Simple metrics function ---
def compute_metrics(y_true, y_pred, num_classes=10):
    y_true = y_true.flatten().cpu().numpy()
    y_pred = y_pred.flatten().cpu().numpy()
    acc = np.mean(y_true == y_pred)
    return {"val_acc": acc}

# --- Fast training with full points ---
def fast_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load real dataset
    dataset = ParisLilleDataset(data_dir='preprocessed_validation_data')

    # Use a small subset of samples (e.g., 8–10) but keep full points
    subset_size = min(100, len(dataset))
    subset_indices = list(range(subset_size))
    dataset = Subset(dataset, subset_indices)

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Load your real model
    num_classes = 10
    model = get_model(num_classes=num_classes, in_channel=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load label weights
    label_weights_path = r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Scripts\preprocessed_validation_data\labelweights.npy"
    if os.path.exists(label_weights_path):
        label_weights = np.load(label_weights_path)
        label_weights_tensor = torch.tensor(label_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=label_weights_tensor)
    else:
        print("Label weights file not found, using unweighted CrossEntropyLoss.")
        criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            preds, _ = model(points)
            loss = criterion(preds.reshape(-1, num_classes), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Train Loss: {total_loss/len(train_loader):.4f}")

        # ---- Validation ----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                preds, _ = model(points)
                all_preds.append(preds.argmax(dim=2))
                all_labels.append(labels)
        metrics = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))
        print(f"Epoch {epoch} Val Acc: {metrics['val_acc']:.4f}")

if __name__ == "__main__":
    fast_train()
