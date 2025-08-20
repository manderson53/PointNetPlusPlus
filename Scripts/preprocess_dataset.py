import os
import numpy as np
from pyntcloud import PyntCloud

def process_ply_file(ply_path, output_dir, file_prefix, block_size=4096):
    cloud = PyntCloud.from_file(ply_path)
    points = cloud.points[["x", "y", "z"]].values
    #points = cloud.points[["x", "y", "z", "reflectance"]].values   # turn this on if you want reflectance data and store it in a new directory
    labels = cloud.points["class"].values
    print("Unique labels in this file:", np.unique(labels))

    print(f"[INFO] Loaded {points.shape[0]} points from {os.path.basename(ply_path)}")

    num_points = points.shape[0]
    block_idx = 0

    # Randomly sample blocks of fixed size
    while block_idx * block_size < num_points:
        start = block_idx * block_size
        end = start + block_size

        if end > num_points:
            # If remaining is less than block size, sample with replacement
            idx = np.random.choice(num_points, block_size, replace=True)
        else:
            idx = np.arange(start, end)

        sampled_points = points[idx]
        sampled_labels = labels[idx]

        np.save(os.path.join(output_dir, f"{file_prefix}_points_{block_idx:04d}.npy"), sampled_points)
        np.save(os.path.join(output_dir, f"{file_prefix}_labels_{block_idx:04d}.npy"), sampled_labels)

        block_idx += 1

    print(f"[DONE] Saved {block_idx} chunks to {output_dir}")

# === Configure this part ===
# PLY_FILES = [
#     r"C:\Users\mikeb\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille1_1.ply",
#     r"C:\Users\mikeb\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille1_2.ply",
#     r"C:\Users\mikeb\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\paris.ply"
# ]
PLY_FILES = [
    r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille1_1.ply",
    r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille1_2.ply",
    r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\paris.ply"
]
OUTPUT_DIR = "preprocessed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
for ply_file in PLY_FILES:
    file_prefix = os.path.splitext(os.path.basename(ply_file))[0]
    process_ply_file(ply_file, OUTPUT_DIR, file_prefix)