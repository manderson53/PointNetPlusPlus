import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from pointnetplusplus import PointNet2Seg


def check_labels(dataset):
    """Check unique labels in the dataset and their distribution."""
    all_labels = []
    for i in range(min(len(dataset), 200)):  # scan first 200 samples for speed
        _, labels = dataset[i]
        all_labels.extend(labels.numpy().ravel().tolist())

    unique_labels = np.unique(all_labels)
    counts = Counter(all_labels)

    print("✅ Unique labels found:", unique_labels)
    print("📊 Label distribution (first 200 samples):")
    for lbl in sorted(unique_labels):
        print(f"  Label {lbl}: {counts[lbl]} points")


def overfit_tiny_batch(dataset, num_classes=10):
    """Try to overfit a tiny batch and see if loss drops below 2.3."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    points, labels = next(iter(loader))

    # normalize
    points = points - points.mean(dim=1, keepdim=True)
    norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
    points = points / norms
    points = points.permute(0, 2, 1).to(device)
    labels = labels.to(device)

    print("\nBatch sanity check:")
    print("  Points shape:", points.shape)
    print("  Labels shape:", labels.shape)
    print("  Unique labels in batch:", torch.unique(labels))

    model = PointNet2Seg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nOverfitting tiny batch...")
    for step in range(50):
        optimizer.zero_grad()
        outputs = model(points).permute(0, 2, 1)
        loss = criterion(outputs.contiguous().view(-1, num_classes), labels.view(-1))
        loss.backward()
        optimizer.step()
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}, Loss: {loss.item():.4f}")

    print("\n👉 If loss falls below ~2.3, labels are okay. If not, check distribution/imbalance.")


if __name__ == "__main__":
    # Load dataset
    dataset = CombinedLille1ParisDataset(data_dirs=["preprocessed_data"])

    print("Checking labels and distribution...")
    check_labels(dataset)

    print("\nRunning tiny batch overfit test...")
    overfit_tiny_batch(dataset)
