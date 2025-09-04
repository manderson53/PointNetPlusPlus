import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

# ----------------- Config -----------------
PLY_FILES = [
    r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille2.ply"
]
OUTPUT_DIR = "preprocessed_validation_data"
BLOCK_SIZE = 4096       # points per block
BLOCK_LENGTH = 1.0      # meters

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

def process_ply_file(ply_path, output_dir, block_size=BLOCK_SIZE, block_length=BLOCK_LENGTH, print_every=100):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    # Shift coordinates to handle negatives
    min_coords = points.min(axis=0)
    points_shifted = points - min_coords

    # Try to get labels (if exist)
    try:
        import pyntcloud
        cloud = pyntcloud.PyntCloud.from_file(ply_path)
        labels = cloud.points["class"].values.astype(np.int32)
        print(f"[INFO] Loaded {points.shape[0]} points with labels from {os.path.basename(ply_path)}")
    except (ImportError, KeyError):
        labels = np.zeros(len(points), dtype=np.int32)
        print(f"[INFO] Loaded {points.shape[0]} points without labels from {os.path.basename(ply_path)}")

    # Compute voxel indices
    voxel_indices = np.floor(points_shifted / block_length).astype(np.int32)
    unique_voxels = np.unique(voxel_indices, axis=0)

    block_idx = 0
    all_labels_for_weights = []

    for vx, vy, vz in tqdm(unique_voxels, desc="Blocks"):
        mask = (
            (voxel_indices[:,0] == vx) &
            (voxel_indices[:,1] == vy) &
            (voxel_indices[:,2] == vz)
        )
        block_points = points[mask]
        block_labels = labels[mask]

        if len(block_points) == 0:
            continue

        # Sample fixed number of points
        if len(block_points) >= block_size:
            idx = np.random.choice(len(block_points), block_size, replace=False)
        else:
            idx = np.random.choice(len(block_points), block_size, replace=True)

        sampled_points = block_points[idx]
        sampled_labels = block_labels[idx]

        # Normalize points (centroid at origin, scale to unit sphere)
        centroid = sampled_points.mean(axis=0)
        normalized_points = sampled_points - centroid
        scale = np.max(np.linalg.norm(normalized_points, axis=1))
        normalized_points = normalized_points / (scale + 1e-6)

        # Save
        prefix = os.path.splitext(os.path.basename(ply_path))[0]
        np.save(os.path.join(output_dir, f"{prefix}_points_{block_idx:04d}.npy"), normalized_points.astype(np.float32))
        np.save(os.path.join(output_dir, f"{prefix}_points_orig_{block_idx:04d}.npy"), sampled_points.astype(np.float32))
        np.save(os.path.join(output_dir, f"{prefix}_labels_{block_idx:04d}.npy"), sampled_labels.astype(np.int32))

        all_labels_for_weights.append(sampled_labels)
        block_idx += 1

        # if block_idx % print_every == 0:
        #     print(f"[INFO] Saved {block_idx} blocks")

    print(f"[DONE] Saved {block_idx} blocks from {os.path.basename(ply_path)}")
    return all_labels_for_weights

# ----------------- Main Processing -----------------
all_labels = []
for ply_file in PLY_FILES:
    labels_per_file = process_ply_file(ply_file, OUTPUT_DIR)
    all_labels.extend(labels_per_file)

# ----------------- Compute label weights -----------------
NUM_CLASSES = 10
if len(all_labels) > 0:
    label_weights = compute_label_weights(all_labels, NUM_CLASSES)
    print("Label weights:", label_weights)
    np.save(os.path.join(OUTPUT_DIR, "labelweights.npy"), label_weights.astype(np.float32))
else:
    print("[INFO] No labels found; skipping label weights computation.")
