import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import os
import numpy as np
from collections import Counter

from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from pointnetplusplus import PointNet2Seg

def scan_dataset_labels(data_dirs=["preprocessed_data"]):
    dataset = CombinedLille1ParisDataset(data_dirs=data_dirs)
    total_counts = Counter()
    file_issues = []

    for idx, (points, labels) in enumerate(dataset):
        labels_np = labels.numpy().ravel()
        counts = Counter(labels_np)
        total_counts.update(counts)

        # Check for unexpected labels (outside 0..9)
        for lbl in counts.keys():
            if lbl < 0 or lbl > 9:
                file_path = dataset.samples[idx][1]
                file_issues.append((file_path, lbl))

    print("✅ Total label counts across dataset:")
    for lbl in sorted(total_counts.keys()):
        print(f"  Label {lbl}: {total_counts[lbl]} points")

    if file_issues:
        print("\n⚠️ Found unexpected labels in the following files:")
        for fpath, lbl in file_issues:
            print(f"  File: {fpath}, Label: {lbl}")
    else:
        print("\n✅ All labels within expected range 0..9")

if __name__ == "__main__":
    scan_dataset_labels()