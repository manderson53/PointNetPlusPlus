from pointnetplusplus import PointNet2Seg
from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import os
import csv
import datetime
from collections import Counter
import random


# Variables
csv_file_path = "validation_metrics.csv"
log_file_path = "validation_metrics.txt"
num_epochs = 10
using_physics_loss = False

CLASS_NAMES = [
    "Unclassified",       # 0
    "Ground",             # 1
    "Building",           # 2
    "Pole",               # 3
    "Bollard",            # 4
    "Trash",              # 5
    "Barrier",            # 6
    "Pedestrian",         # 7
    "Car",                # 8
    "Vegitation"          # 9
]

# Uncomment and use if deterministic behavior is needed
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

def compute_class_counts(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in dataset:
        labels_np = labels.numpy().flatten()
        label_counts = Counter(labels_np)
        for cls, cnt in label_counts.items():
            counts[cls] += cnt
    return counts

def balanced_sample_with_oversample(points, labels, samples_per_class=100, jitter_std=0.01):
    unique_labels = labels.unique()
    point_list = []
    label_list = []

    for ul in unique_labels:
        inds = (labels == ul).nonzero(as_tuple=True)[0]
        n = len(inds)
        if n == 0:
            continue

        repeat_factor = (samples_per_class + n - 1) // n
        repeated_inds = inds.repeat(repeat_factor)[:samples_per_class]

        pts = points[repeated_inds]
        lbs = labels[repeated_inds]

        # Apply jitter augmentation
        jitter = torch.randn_like(pts) * jitter_std
        pts = pts + jitter

        point_list.append(pts)
        label_list.append(lbs)

    balanced_points = torch.cat(point_list, dim=0)
    balanced_labels = torch.cat(label_list, dim=0)
    return balanced_points, balanced_labels

def prepare_balanced_batch(points_batch, labels_batch, min_samples_per_class=100, jitter_std=0.01):
    balanced_points_list = []
    balanced_labels_list = []

    for i in range(points_batch.size(0)):
        pts = points_batch[i]
        lbs = labels_batch[i]
        unique_labels = lbs.unique()

        # Get counts per class in this sample
        counts = torch.tensor([(lbs == ul).sum().item() for ul in unique_labels])
        min_points = counts.min().item()

        # Use the max between min_samples_per_class and min_points
        sample_size = max(min_points, min_samples_per_class)

        b_pts, b_lbs = balanced_sample_with_oversample(pts, lbs, samples_per_class=sample_size, jitter_std=jitter_std)
        balanced_points_list.append(b_pts)
        balanced_labels_list.append(b_lbs)

    max_points = max([bp.shape[0] for bp in balanced_points_list])
    padded_points = []
    padded_labels = []

    for b_pts, b_lbs in zip(balanced_points_list, balanced_labels_list):
        pad_len = max_points - b_pts.shape[0]
        if pad_len > 0:
            pad_pts = torch.zeros(pad_len, 3, device=b_pts.device)
            pad_lbs = torch.full((pad_len,), -1, dtype=b_lbs.dtype, device=b_lbs.device)
            b_pts = torch.cat([b_pts, pad_pts], dim=0)
            b_lbs = torch.cat([b_lbs, pad_lbs], dim=0)
        padded_points.append(b_pts)
        padded_labels.append(b_lbs)

    return torch.stack(padded_points), torch.stack(padded_labels)


dataset = CombinedLille1ParisDataset(data_dirs=["preprocessed_data"])
num_classes = 10
class_counts = compute_class_counts(dataset, num_classes)

# Avoid zero counts to prevent division errors
class_counts = np.maximum(class_counts, 1)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

#remove weighting since we're oversampling the minimum points and balancing class distribution
# weights = torch.tensor(class_weights, dtype=torch.float32)
weights = torch.ones(num_classes, dtype=torch.float32)  # equal weights for all classes


def train_kfold(k=5):
    # Initialize CSV with headers once
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["timestamp", "fold", "epoch", "mean_iou"] + [f"{CLASS_NAMES[i]} ({i})" for i in range(len(CLASS_NAMES))] + ["val_cross_entropy_loss", "val_physics_loss"]
        writer.writerow(header)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "preprocessed_data"
    folds = get_folds(data_dir, k=k)

    for fold, (train_files, val_files) in enumerate(folds):
        print(f"Starting fold {fold+1}/{k}")

        train_dataset = CombinedLille1ParisDataset(files_lists=[train_files])
        val_dataset = CombinedLille1ParisDataset(files_lists=[val_files])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

        model = PointNet2Seg(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)  # Ignore padded labels
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_miou = -float('inf')  # Track best mean IoU

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for points, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}"):
                #balanced_points_batch, balanced_labels_batch = prepare_balanced_batch(points, labels, samples_per_class=100, jitter_std=0.01)
                balanced_points_batch, balanced_labels_batch = prepare_balanced_batch(points, labels, jitter_std=0.01)

                #debug logging
                # print("==== Debug Batch Info ====")
                # print(f"Original points shape: {points.shape}")  # (B, N, 3)
                # print(f"Original labels shape: {labels.shape}")  # (B, N)
                
                # print(f"Balanced points batch shape: {balanced_points_batch.shape}")  # (B, M, 3)
                # print(f"Balanced labels batch shape: {balanced_labels_batch.shape}")  # (B, M)
                
                # unique_labels, counts = torch.unique(balanced_labels_batch, return_counts=True)
                # print(f"Balanced labels unique classes: {unique_labels.tolist()}")
                # print(f"Per-class counts in balanced batch: {counts.tolist()}")
                
                # Check padded labels presence (-1)
                # num_padded = (balanced_labels_batch == -1).sum().item()
                # print(f"Number of padded points in batch: {num_padded}")


                # Normalize and permute for model input
                balanced_points_batch = balanced_points_batch - balanced_points_batch.mean(dim=1, keepdim=True)
                norms = balanced_points_batch.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
                balanced_points_batch = balanced_points_batch / norms
                balanced_points_batch = balanced_points_batch.permute(0, 2, 1).to(device)  # (B, 3, N)
                balanced_labels_batch = balanced_labels_batch.to(device)

                optimizer.zero_grad()
                outputs = model(balanced_points_batch)  # (B, N, num_classes)
                outputs = outputs.permute(0, 2, 1)      # (B, num_classes, N)
                outputs_flat = outputs.contiguous().view(-1, num_classes)
                labels_flat = balanced_labels_batch.view(-1)

                # Print output logits stats
                # print(f"Outputs_flat shape: {outputs_flat.shape}")
                # print(f"Outputs_flat min: {outputs_flat.min().item()}, max: {outputs_flat.max().item()}, mean: {outputs_flat.mean().item()}")

                #print out logits
                # unique_labels_in_flat, counts_in_flat = torch.unique(labels_flat, return_counts=True)
                # print(f"Labels_flat unique classes: {unique_labels_in_flat.tolist()}")
                # print(f"Labels_flat per-class counts: {counts_in_flat.tolist()}")

                if using_physics_loss:
                    loss_ce = criterion(outputs_flat, labels_flat)
                    pred_labels = outputs.argmax(dim=2)
                    loss_phys = physics_loss(pred_labels, balanced_points_batch)
                    loss = loss_ce + 0.1 * loss_phys
                else:
                    loss = criterion(outputs_flat, labels_flat)

                #print out loss value
                # loss = criterion(outputs_flat, labels_flat)
                # print(f"Loss value: {loss.item()}")

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Fold {fold+1} Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

            # Validation phase
            model.eval()
            all_ious = []
            all_ce_losses = []
            all_phys_losses = []

            with torch.no_grad():
                for points, labels in val_loader:
                    points = points.permute(0, 2, 1).to(device)
                    labels = labels.to(device)

                    outputs = model(points)
                    outputs = outputs.permute(0, 2, 1)
                    outputs_flat = outputs.contiguous().view(-1, num_classes)
                    labels_flat = labels.view(-1)
                    preds = outputs.argmax(dim=1)

                    loss_ce = criterion(outputs_flat, labels_flat)
                    all_ce_losses.append(loss_ce.item())

                    if using_physics_loss:
                        pred_labels = outputs.argmax(dim=2)
                        loss_phys = physics_loss(pred_labels, points)
                        all_phys_losses.append(loss_phys.item())

                    batch_ious = []
                    for b in range(preds.shape[0]):
                        pred_sample = preds[b].cpu()
                        label_sample = labels[b].cpu()
                        ious = compute_iou(pred_sample, label_sample)
                        batch_ious.append(ious)

                    batch_ious = np.array(batch_ious)
                    all_ious.extend(batch_ious)

            mean_ce_loss = np.mean(all_ce_losses)
            if using_physics_loss:
                mean_physics_loss = np.nanmean(all_phys_losses)
            else:
                mean_physics_loss = float('nan')

            all_ious = np.array(all_ious)
            mean_class_ious = np.nanmean(all_ious, axis=0)
            mean_miou = np.nanmean(mean_class_ious)

            print(f"Fold {fold+1} Epoch {epoch+1} Validation mIoU: {mean_miou:.4f}, CE loss: {mean_ce_loss:.4f}, Physics loss: {mean_physics_loss:.4f}")
            print("Per-class IoU:")
            for i, iou in enumerate(mean_class_ious):
                class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
                print(f"  {class_name} ({i}): {iou:.4f}")

            # Save metrics to CSV
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = [datetime.datetime.now(), fold+1, epoch+1, mean_miou] + mean_class_ious.tolist() + [mean_ce_loss, mean_physics_loss]
                writer.writerow(row)

            # Save metrics to text log
            with open(log_file_path, "a") as f:
                f.write(f"{datetime.datetime.now()} - Fold {fold+1} Epoch {epoch+1} Validation mIoU: {mean_miou:.4f}, CE loss: {mean_ce_loss:.4f}, Physics loss: {mean_physics_loss:.4f}\n")
                f.write("Per-class IoU:\n")
                for i, iou in enumerate(mean_class_ious):
                    class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
                    f.write(f"  {class_name} ({i}): {iou:.4f}\n")
                f.write("\n")

            # Save best model per fold
            if mean_miou > best_miou:
                best_miou = mean_miou
                model_save_path = f"pointnet2_parislille_fold{fold+1}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved best model for fold {fold+1} at epoch {epoch+1} with mIoU {best_miou:.4f}")


def physics_loss(pred_labels, points):
    z = torch.nan_to_num(points[:, 2, :], nan=0.0)
    penalty = torch.zeros_like(z, dtype=torch.float32)

    # Road (class 1) penalty for points above 2 meters
    road_mask = (pred_labels == 1)
    penalty += road_mask * (z - 2.0).clamp(min=0.0)

    # Building (class 3) penalty for points below 1 meter
    building_mask = (pred_labels == 3)
    penalty += building_mask * (1.0 - z).clamp(min=0.0)

    # People (class 8) penalty if below 0 or above 2 meters
    people_mask = (pred_labels == 8)
    penalty += people_mask * ((0.0 - z).clamp(min=0.0) + (z - 2.0).clamp(min=0.0))

    # Cars (class 7) penalty if below 0 or above 3 meters
    cars_mask = (pred_labels == 7)
    penalty += cars_mask * ((0.0 - z).clamp(min=0.0) + (z - 3.0).clamp(min=0.0))

    # Trash cans (class 5) penalty for points above 2 meters
    trash_mask = (pred_labels == 5)
    penalty += trash_mask * (z - 2.0).clamp(min=0.0)

    return penalty.mean()


def get_folds(data_dir, k=5):
    all_label_files = sorted([f for f in os.listdir(data_dir) if "_labels_" in f and f.endswith(".npy")])
    all_point_files = sorted([f for f in os.listdir(data_dir) if "_points_" in f and f.endswith(".npy")])

    # Sanity check
    assert len(all_label_files) == len(all_point_files)

    all_label_paths = [os.path.join(data_dir, f) for f in all_label_files]
    all_point_paths = [os.path.join(data_dir, f) for f in all_point_files]

    samples = list(zip(all_point_paths, all_label_paths))

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    folds = []
    samples = np.array(samples)
    for train_idx, val_idx in kf.split(samples):
        train_samples = samples[train_idx].tolist()
        val_samples = samples[val_idx].tolist()
        folds.append((train_samples, val_samples))

    return folds


def compute_iou(preds, labels, num_classes=10):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious


if __name__ == "__main__":
    train_kfold()
