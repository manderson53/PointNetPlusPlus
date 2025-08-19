from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from pointnetplusplus import PointNet2Seg
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# ----------------- Config -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
batch_size = 32 #was 8
num_epochs = 100
data_dir = "preprocessed_data"
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# ----------------- Class Weights -----------------
counts = torch.tensor([1462643, 53034315, 25824813, 546275, 54421,
                       180665, 3997055, 160917, 3442269, 9682547], dtype=torch.float32)
weights = 1.0 / counts
weights = weights / weights.sum() * len(counts)
weights = weights.to(device)

def train_model():
    # ----------------- Dataset -----------------
    dataset = CombinedLille1ParisDataset(data_dirs=[data_dir])
    print("Full dataset size:", len(dataset))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # ----------------- Model, Loss, Optimizer -----------------
    model = PointNet2Seg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ----------------- CSV Logging -----------------
    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss"])

    # ----------------- Training Loop -----------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Normalize points
            points = points - points.mean(dim=1, keepdim=True)
            norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
            points = points / norms
            points = points.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            # Forward + Backward
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * points.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Save metrics
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss])

        # Save model
        save_path = os.path.join(save_dir, f"{epoch+1}_pointnet2.pth")
        torch.save(model.state_dict(), save_path)

        print(f"\nEpoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f}")
        print(f"✅ Saved model at epoch {epoch+1}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()