import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Lille2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dirs=None, files_lists=None):
        """
        data_dirs: List of data directories (strings), e.g., ["lille1_1_dir", "lille2_dir", "paris_dir"]
        files_lists: Optional; list of lists of (points_file, labels_file) tuples for each dataset
        """
        self.samples = []
        if files_lists is not None:
            # Combine file lists explicitly provided
            for flist in files_lists:
                self.samples.extend(flist)
        elif data_dirs is not None:
            # Load files from each directory and append
            for d in data_dirs:
                label_files = sorted([f for f in os.listdir(d) if "_labels_" in f and f.endswith(".npy")])
                point_files = sorted([f for f in os.listdir(d) if "_points_" in f and f.endswith(".npy")])
                samples_in_dir = [(os.path.join(d, p), os.path.join(d, l)) for p, l in zip(point_files, label_files)]
                self.samples.extend(samples_in_dir)
        else:
            raise ValueError("Either data_dirs or files_lists must be provided.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        point_path, label_path = self.samples[idx]
        points = np.load(point_path)   # should be shape (N,3)
        labels = np.load(label_path)   # should be shape (N,)

        if points is None or labels is None:
            print(f"Warning: None returned for index {idx}")
        
        xyz = torch.from_numpy(points).float().permute(1, 0)  # (3, N)
        labels = torch.from_numpy(labels).long()              # (N,)
        return xyz, labels
