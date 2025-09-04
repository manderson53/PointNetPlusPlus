import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ParisLilleDataset(Dataset):
    def __init__(self, data_dir, transform=None, load_orig=False):
        """
        Args:
            data_dir (str): Directory with preprocessed .npy files.
            transform (callable, optional): Function to apply augmentations to points and labels.
            load_orig (bool, optional): If True, also return original points.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.load_orig = load_orig

        # Collect and match points and labels
        all_files = os.listdir(data_dir)
        self.points_files = sorted([f for f in all_files 
                            if 'points_' in f and 'points_orig' not in f and f.endswith('.npy')])
        self.labels_files = sorted([f for f in all_files 
                            if 'labels_' in f and f.endswith('.npy')])
        if load_orig:
            self.orig_files = sorted([f for f in all_files 
                                      if 'points_orig_' in f and f.endswith('.npy')])
            assert len(self.orig_files) == len(self.points_files), "Mismatch between points and original points!"
        else:
            self.orig_files = [None] * len(self.points_files)
            assert len(self.points_files) == len(self.labels_files), "Mismatch between points and labels files! point files total: " + str(len(self.points_files)) + " label files total: " + str(len(self.labels_files))

        # Load precomputed XYZ ranges
        xyz_file = os.path.join(
            data_dir,
            'block_xyz_ranges.npy'
        )
        self.block_xyz_range = np.load(xyz_file, allow_pickle=True).item()

    def __len__(self):
        return len(self.points_files)

    def __getitem__(self, idx):
        points_path = os.path.join(self.data_dir, self.points_files[idx])
        labels_path = os.path.join(self.data_dir, self.labels_files[idx])
        orig_path = os.path.join(self.data_dir, self.orig_files[idx]) if self.load_orig else None

        points = np.load(points_path)
        labels = np.load(labels_path)
        points_orig = np.load(orig_path) if self.load_orig else None

        block_name = get_block_name_from_file(self.points_files[idx])
        xyz_min = torch.from_numpy(self.block_xyz_range[block_name]['min']).float()
        xyz_max = torch.from_numpy(self.block_xyz_range[block_name]['max']).float()

        if self.transform:
            points, labels = self.transform(points, labels)

        points = torch.from_numpy(points).float().transpose(0, 1)  # [3, 4096]
        labels = torch.from_numpy(labels).long()
        if self.load_orig:
            points_orig = torch.from_numpy(points_orig).float().transpose(0, 1)
            return points, labels, block_name, xyz_min, xyz_max, points_orig
        else:
            return points, labels, block_name, xyz_min, xyz_max


# --- Augmentation functions ---
def train_augment_full(points, labels):
    points, labels = random_rotation(points, labels)
    points, labels = jitter(points, labels)
    points, labels = random_scale(points, labels)
    points, labels = random_flip_x(points, labels)
    return points, labels

def random_rotation(points, labels=None):
    theta = np.random.uniform(0, 2*np.pi)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    points = points @ rot_matrix.T
    return points, labels

def jitter(points, labels=None, sigma=0.01, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    points = points + noise
    return points, labels

def random_scale(points, labels=None, scale_low=0.9, scale_high=1.1):
    scale = np.random.uniform(scale_low, scale_high)
    points = points * scale
    return points, labels

def random_flip_x(points, labels=None, p=0.5):
    if np.random.rand() < p:
        points[:, 0] = -points[:, 0]
    return points, labels

# --- Helper functions ---
def get_block_name_from_file(filename: str):
    """
    Extract block name from file, ignoring '_points_orig_' files.
    Examples:
        'Lille1_1_points_0000.npy' -> 'Lille1_1'
        'Paris_points_0123.npy'   -> 'Paris'
    """
    base = os.path.basename(filename)
    # if "_points_orig_" in base:
    #     return None  # skip original points
    for key in ['_points_', '_labels_']:
        if key in base:
            return base.split(key)[0]
    raise ValueError(f"Cannot determine block name from {filename}")
