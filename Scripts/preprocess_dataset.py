import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pyntcloud import PyntCloud

try:
    from config_local import TRAINING_PLY_FILES as PLY_FILES, PREPROCESS_TRAINING_DATA_OUTPUT_DIR as OUTPUT_DIR
except ImportError:
    raise RuntimeError("Missing config_local.py. Please create it with PLY_FILES and OUTPUT_DIR defined.")

# ----------------- Config -----------------
BLOCK_SIZE = 4096       # points per block
BLOCK_LENGTH = 1.0      # meters (1x1x1 m blocks)
OVERLAP_XY = 0.2   # 20% in X and Y
OVERLAP_Z  = 0.05  # 5% in Z
NUM_CLASSES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Helper Functions -----------------
def compute_label_weights(all_labels, num_classes):
    counts = np.zeros(num_classes)
    for l in all_labels:
        counts += np.bincount(l, minlength=num_classes)
    freq = counts / counts.sum()
    weights = 1 / (freq + 1e-6)
    weights = weights / weights.min()  # normalize so min weight = 1
    return weights

def process_ply_file(ply_path):
    # Load point cloud and labels
    cloud = PyntCloud.from_file(ply_path)
    points = cloud.points[["x", "y", "z"]].values
    labels = cloud.points["class"].values.astype(np.int32)

    print(f"[INFO] Loaded {points.shape[0]} points from {os.path.basename(ply_path)}")
    print("Unique labels in file:", np.unique(labels))

    # Shift coordinates so min = 0
    min_coords = points.min(axis=0)
    points_shifted = points - min_coords

    # Voxel indices with 50% overlap
    # step = BLOCK_LENGTH * (1 - OVERLAP)
    step_x = BLOCK_LENGTH * (1 - OVERLAP_XY)
    step_y = BLOCK_LENGTH * (1 - OVERLAP_XY)
    step_z = BLOCK_LENGTH * (1 - OVERLAP_Z)
    vx = np.floor(points_shifted[:,0] / step_x).astype(np.int32)
    vy = np.floor(points_shifted[:,1] / step_y).astype(np.int32)
    vz = np.floor(points_shifted[:,2] / step_z).astype(np.int32)
    voxel_indices = np.stack([vx, vy, vz], axis=1)
    # voxel_indices = np.floor(points_shifted / step).astype(np.int32)
    unique_voxels = np.unique(voxel_indices, axis=0)

    block_idx = 0
    all_labels_for_weights = []
    prefix = os.path.splitext(os.path.basename(ply_path))[0]

    # One progress bar per file
    for vx_i, vy_i, vz_i in tqdm(unique_voxels, desc=f"Processing {prefix}"):
        mask = (
            (points_shifted[:,0] >= vx_i*step_x) & (points_shifted[:,0] < (vx_i+1)*step_x) &
            (points_shifted[:,1] >= vy_i*step_y) & (points_shifted[:,1] < (vy_i+1)*step_y) &
            (points_shifted[:,2] >= vz_i*step_z) & (points_shifted[:,2] < (vz_i+1)*step_z)
        )
        block_points = points[mask]
        block_labels = labels[mask]

        if len(block_points) == 0:
            continue

        # Sample fixed number of points
        if len(block_points) >= BLOCK_SIZE:
            idx = np.random.choice(len(block_points), BLOCK_SIZE, replace=False)
        else:
            idx = np.random.choice(len(block_points), BLOCK_SIZE, replace=True)

        sampled_points = block_points[idx]
        sampled_labels = block_labels[idx]

        # Normalize points (centroid at origin, scale to unit sphere)
        centroid = sampled_points.mean(axis=0)
        normalized_points = sampled_points - centroid
        scale = np.max(np.linalg.norm(normalized_points, axis=1))
        normalized_points = normalized_points / (scale + 1e-6)

        # Save
        np.save(os.path.join(OUTPUT_DIR, f"{prefix}_points_{block_idx:04d}.npy"), normalized_points.astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"{prefix}_points_orig_{block_idx:04d}.npy"), sampled_points.astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"{prefix}_labels_{block_idx:04d}.npy"), sampled_labels.astype(np.int32))

        all_labels_for_weights.append(sampled_labels)
        block_idx += 1

    print(f"[DONE] Saved {block_idx} blocks from {os.path.basename(ply_path)}")
    return all_labels_for_weights

# ----------------- Main Processing -----------------
all_labels = []

for ply_file in PLY_FILES:
    labels_per_file = process_ply_file(ply_file)
    all_labels.extend(labels_per_file)

# ----------------- Compute label weights -----------------
if len(all_labels) > 0:
    label_weights = compute_label_weights(all_labels, NUM_CLASSES)
    print("Label weights:", label_weights)
    np.save(os.path.join(OUTPUT_DIR, "labelweights.npy"), label_weights.astype(np.float32))
else:
    print("[INFO] No labels found; skipping label weights computation.")

print("[DONE] Training preprocessing complete.")
