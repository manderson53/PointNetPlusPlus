import torch
import torch.nn as nn
from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from pointnetplusplus import PointNet2Seg
import time

class DummyLinear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(3, num_classes)

    def forward(self, x):
        batch_size, channels, num_points = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 3)  # (B*N, 3)
        out = self.linear(x)                    # (B*N, num_classes)
        out = out.view(batch_size, num_points, -1)
        return out.permute(0, 2, 1)            # (B, num_classes, N)

def balanced_sample(points, labels, samples_per_class=100):
    unique_labels = labels.unique()
    selected_indices = []
    print(f"  [balanced_sample] unique classes: {unique_labels.tolist()}")
    for ul in unique_labels:
        inds = (labels == ul).nonzero(as_tuple=True)[0]
        if len(inds) < samples_per_class:
            sampled = inds
            print(f"    Class {ul.item()} has only {len(inds)} points, using all")
        else:
            sampled = inds[torch.randperm(len(inds))[:samples_per_class]]
            print(f"    Class {ul.item()} sampled {samples_per_class} points")
        selected_indices.append(sampled)
    balanced_inds = torch.cat(selected_indices)
    print(f"  [balanced_sample] total balanced points: {len(balanced_inds)}")
    return points[balanced_inds], labels[balanced_inds]

def train_debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    print("[train_debug] Loading dataset...")
    dataset = CombinedLille1ParisDataset(data_dirs=["preprocessed_data"])
    print(f"[train_debug] Dataset size: {len(dataset)}")

    idx = 2  # sample index with multiple classes
    print(f"[train_debug] Loading sample #{idx} for balanced sampling")
    points, labels = dataset[idx]
    print(f"[train_debug] Sample points shape: {points.shape}")
    print(f"[train_debug] Sample labels unique classes: {labels.unique().tolist()}")

    # Balanced sampling
    print("[train_debug] Performing balanced sampling...")
    balanced_points, balanced_labels = balanced_sample(points, labels, samples_per_class=100)
    print(f"[train_debug] Balanced points shape: {balanced_points.shape}")
    print(f"[train_debug] Balanced labels shape: {balanced_labels.shape}")

    # Add batch dimension and move to device
    balanced_points = balanced_points.unsqueeze(0).to(device)
    balanced_labels = balanced_labels.unsqueeze(0).to(device)

    # Normalize points
    print("[train_debug] Normalizing points...")
    balanced_points = balanced_points - balanced_points.mean(dim=1, keepdim=True)
    norms = balanced_points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
    balanced_points = balanced_points / norms
    balanced_points = balanced_points.permute(0, 2, 1)  # (B, 3, N)
    print(f"[train_debug] Normalized points shape: {balanced_points.shape}")

    model = PointNet2Seg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_epochs = 50

    for epoch in range(num_epochs):
        print(f"[train_debug] Epoch {epoch+1}/{num_epochs} - start")
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(balanced_points)  # (B, N, num_classes)
        
        print(f"[train_debug] Forward pass completed")
        
        outputs_flat = outputs.view(-1, num_classes)
        labels_flat = balanced_labels.view(-1)
        
        loss = criterion(outputs_flat, labels_flat)
        print(f"[train_debug] Loss computed: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        print(f"[train_debug] Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")
        preds = outputs.argmax(dim=2)
        print("[train_debug] Predicted labels (first 10):", preds[0, :10].cpu().tolist())
        print("[train_debug] True labels (first 10):", balanced_labels[0, :10].cpu().tolist())

if __name__ == "__main__":
    train_debug()
