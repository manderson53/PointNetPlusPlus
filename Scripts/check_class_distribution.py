import os
import numpy as np
from collections import Counter

CLASS_NAMES = [
    "Unclassified",        # 0
    "Ground",              # 1
    "Building",            # 2
    "Pole",                # 3
    "Bollard",             # 4
    "Trash",               # 5
    "Barrier",             # 6
    "Pedestrian",          # 7
    "Car",                 # 8
    "Vegitation"           # 9
]

# Path to your preprocessed .npy label files
LABEL_DIR = "preprocessed_data"

# Find all .npy label files
label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith("_labels_0000.npy") or "_labels_" in f]

# Initialize a Counter to count all class labels
label_counts = Counter()

for fname in label_files:
    path = os.path.join(LABEL_DIR, fname)
    labels = np.load(path)
    flat = labels.flatten().tolist()
    label_counts.update(flat)

# Print the total number of points per class
total = sum(label_counts.values())
output_path = "class_distribution.txt"

with open(output_path, "w") as f:
    f.write("Class distribution:\n")
    print("\nClass distribution:")
    for c in range(10):
        count = label_counts.get(c, 0)
        pct = 100.0 * count / total if total > 0 else 0
        class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
        line = f"{class_name} ({c}): {count:,} points ({pct:.2f}%)"
        print(line)
        f.write(line + "\n")

    total_line = f"\nTotal points: {total:,}"
    print(total_line)
    f.write(total_line + "\n")
