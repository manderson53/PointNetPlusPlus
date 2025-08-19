from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from pointnetplusplus import PointNet2Seg
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import csv
import numpy as np

# ----------------- Config -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
batch_size = 32
num_epochs = 10  # shorter for subset test
train_split = 0.8
data_dir = "preprocessed_data"
save_dir = "checkpoints_subset"
os.makedirs(save_dir, exist_ok=True)

# ----------------- Class Weights -----------------
counts = torch.tensor([1462643, 53034315, 25824813, 546275, 54421,
                       180665, 3997055, 160917, 3442269, 9682547], dtype=torch.float32)
weights = 1.0 / counts
weights = weights / weights.sum() * len(counts)
weights = weights.to(device)

def train_subset(subset_size=500):
    # ----------------- Dataset -----------------
    full_dataset = CombinedLille1ParisDataset(data_dirs=[data_dir])
    print("Full dataset size:", len(full_dataset))
    
    # Use only a subset for quick test
    subset_indices = list(range(min(subset_size, len(full_dataset))))
    subset_dataset = Subset(full_dataset, subset_indices)
    
    num_train = int(len(subset_dataset) * train_split)
    num_val = len(subset_dataset) - num_train
    train_dataset, val_dataset = random_split(subset_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ----------------- Model, Loss, Optimizer -----------------
    model = PointNet2Seg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ----------------- Validation -----------------
    def validate(model, loader):
        model.eval()
        total_loss = 0
        correct = 0
        total_points = 0
        num_classes = 10
        intersection = np.zeros(num_classes, dtype=np.int64)
        union = np.zeros(num_classes, dtype=np.int64)
        with torch.no_grad():
            for points, labels in loader:
                points = points - points.mean(dim=1, keepdim=True)
                norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
                points = points / norms
                points = points.permute(0, 2, 1).to(device)
                labels = labels.to(device)

                outputs = model(points)
                loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
                total_loss += loss.item() * points.size(0)

                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total_points += labels.numel()

                # Compute per-class IoU
                for c in range(num_classes):
                    pred_c = (preds == c)
                    label_c = (labels == c)
                    intersection[c] += (pred_c & label_c).sum().item()
                    union[c] += (pred_c | label_c).sum().item()

        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / total_points
        per_class_iou = intersection / (union + 1e-6)
        mean_iou = per_class_iou.mean()
        return avg_loss, accuracy, per_class_iou, mean_iou

    # ----------------- Training Loop -----------------
    best_val_loss = float('inf')
    # Prepare CSV
    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Acc", "Median_IoU"] + [f"IoU_Class_{c}" for c in range(num_classes)])
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            points = points - points.mean(dim=1, keepdim=True)
            norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
            points = points / norms
            points = points.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * points.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, per_class_iou, median_iou = validate(model, val_loader)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, val_acc, median_iou] + per_class_iou.tolist())

        print(f"\nEpoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f"{epoch+1}_pointnet2_subset_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {epoch+1}")

    print("Subset training complete.")

if __name__ == "__main__":
    train_subset(subset_size=10000)  # change subset_size as needed
    # train_subset(None)
