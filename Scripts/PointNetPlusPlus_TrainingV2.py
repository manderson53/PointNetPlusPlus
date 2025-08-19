import torch
import torch.nn as nn
from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from pointnetplusplus import PointNet2Seg
import time
from sklearn.model_selection import KFold
import numpy as np
import os
from torch.utils.data import Subset, DataLoader
import datetime

# ---------- Utility functions ----------

logfile = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_pointnetPlusPlus_Training_V2.txt")

def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(logfile, "a") as f:
        f.write(line + "\n")


def get_folds(data_dir, k=5):
    all_label_files = sorted([f for f in os.listdir(data_dir) if "_labels_" in f and f.endswith(".npy")])
    all_point_files = sorted([f for f in os.listdir(data_dir) if "_points_" in f and f.endswith(".npy")])
    samples = list(zip(all_point_files, all_label_files))
    samples = np.array(samples)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in kf.split(samples):
        train_samples = [(os.path.join(data_dir, p), os.path.join(data_dir, l)) for p, l in samples[train_idx]]
        val_samples = [(os.path.join(data_dir, p), os.path.join(data_dir, l)) for p, l in samples[val_idx]]
        folds.append((train_samples, val_samples))
    return folds

def validate(model, val_loader, criterion, device, num_classes=10):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for points, labels in val_loader:
            points = points.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            outputs = model(points)  # (B, N, num_classes)
            outputs = outputs.permute(0, 2, 1)  # (B, num_classes, N)

            loss = criterion(outputs.contiguous().view(-1, num_classes), labels.view(-1))
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)  # (B, N)
            all_predictions.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(val_loader)

    # Concatenate all batches for metric calculation
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute per-class IoU and mean IoU, ignoring NaNs
    per_class_ious = compute_iou(all_predictions, all_labels, num_classes)
    mean_iou = np.nanmean(per_class_ious)

    log_message(f"Validation Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}")
    for i, iou in enumerate(per_class_ious):
        log_message(f"  Class {i} IoU: {iou:.4f}")

    return avg_loss, mean_iou

def compute_iou(preds, labels, num_classes=10):
    ious = []
    preds = preds.numpy()
    labels = labels.numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()
        if union == 0:
            iou = float('nan')  # Ignore this class in average if union is zero
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

def balanced_sample(points, labels, samples_per_class=100):
    unique_labels = labels.unique()
    selected_indices = []
    log_message(f"  [balanced_sample] unique classes: {unique_labels.tolist()}")
    for ul in unique_labels:
        inds = (labels == ul).nonzero(as_tuple=True)[0]
        if len(inds) < samples_per_class:
            sampled = inds
            log_message(f"    Class {ul.item()} has only {len(inds)} points, using all")
        else:
            sampled = inds[torch.randperm(len(inds))[:samples_per_class]]
            log_message(f"    Class {ul.item()} sampled {samples_per_class} points")
        selected_indices.append(sampled)
    balanced_inds = torch.cat(selected_indices)
    log_message(f"  [balanced_sample] total balanced points: {len(balanced_inds)}")
    return points[balanced_inds], labels[balanced_inds]


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

        counts = torch.tensor([(lbs == ul).sum().item() for ul in unique_labels])
        min_points = counts.min().item()

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


# ---------- Training functions ----------

def train_debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    log_message("[train_debug] Loading dataset...")
    dataset = CombinedLille1ParisDataset(data_dirs=["preprocessed_data"])
    log_message(f"[train_debug] Dataset size: {len(dataset)}")

    idx = 2  # sample index with multiple classes
    log_message(f"[train_debug] Loading sample #{idx} for balanced sampling")
    points, labels = dataset[idx]
    log_message(f"[train_debug] Sample points shape: {points.shape}")
    log_message(f"[train_debug] Sample labels unique classes: {labels.unique().tolist()}")

    log_message("[train_debug] Performing balanced sampling...")
    balanced_points, balanced_labels = balanced_sample(points, labels, samples_per_class=100)
    log_message(f"[train_debug] Balanced points shape: {balanced_points.shape}")
    log_message(f"[train_debug] Balanced labels shape: {balanced_labels.shape}")

    balanced_points = balanced_points.unsqueeze(0).to(device)
    balanced_labels = balanced_labels.unsqueeze(0).to(device)

    log_message("[train_debug] Normalizing points...")
    balanced_points = balanced_points - balanced_points.mean(dim=1, keepdim=True)
    norms = balanced_points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
    balanced_points = balanced_points / norms
    balanced_points = balanced_points.permute(0, 2, 1)  # (B, 3, N)
    log_message(f"[train_debug] Normalized points shape: {balanced_points.shape}")

    model = PointNet2Seg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_epochs = 50

    for epoch in range(num_epochs):
        log_message(f"[train_debug] Epoch {epoch+1}/{num_epochs} - start")
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(balanced_points)  # (B, N, num_classes)
        log_message(f"[train_debug] Forward pass completed")

        outputs_flat = outputs.view(-1, num_classes)
        labels_flat = balanced_labels.view(-1)

        loss = criterion(outputs_flat, labels_flat)
        log_message(f"[train_debug] Loss computed: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

        end_time = time.time()
        log_message(f"[train_debug] Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")
        preds = outputs.argmax(dim=2)
        log_message(f"[train_debug] Predicted labels (first 10): {preds[0, :10].cpu().tolist()}")
        log_message(f"[train_debug] True labels (first 10): {balanced_labels[0, :10].cpu().tolist()}")

def train_partial(sample_size=1000, batch_size=4, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    log_message("[train_partial] Loading full dataset...")
    full_dataset = CombinedLille1ParisDataset(data_dirs=["preprocessed_data"])
    log_message(f"[train_partial] Full dataset size: {len(full_dataset)}")

    # Create dataset subset
    sample_size = min(sample_size, len(full_dataset))
    subset_indices = list(range(sample_size))
    subset_dataset = Subset(full_dataset, subset_indices)
    log_message(f"[train_partial] Subset dataset size: {len(subset_dataset)}")

    loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = PointNet2Seg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    # print(torch.cuda.memory_summary())
    # torch.cuda.empty_cache()

    num_batches = len(loader)

    for epoch in range(num_epochs):
        log_message(f"[train_partial] Epoch {epoch+1}/{num_epochs} - start")
        epoch_start_time = time.time()
        total_loss = 0.0

        for batch_idx, (points, labels) in enumerate(loader):
            points = points.to(device)
            labels = labels.to(device)

            # Normalize points batch: center and scale
            points = points - points.mean(dim=1, keepdim=True)
            norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
            points = points / norms

            # Permute points to (B, 3, N) expected by PointNet++
            points = points.permute(0, 2, 1)

            optimizer.zero_grad()
            outputs = model(points)  # (B, N, num_classes)
            
            outputs_flat = outputs.view(-1, num_classes)
            labels_flat = labels.view(-1)

            loss = criterion(outputs_flat, labels_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                log_message(f"[train_partial] Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start_time
        log_message(f"[train_partial] Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

def train_partial_kfold_new(sample_size=1000, batch_size=8, num_epochs=10, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    log_message("[train_partial_kfold_new] Loading full dataset...")
    full_dataset = CombinedLille1ParisDataset(data_dirs=["preprocessed_data"])
    log_message(f"[train_partial_kfold_new] Full dataset size: {len(full_dataset)}")

    sample_size = min(sample_size, len(full_dataset))
    subset_indices = list(range(sample_size))
    subset_dataset = Subset(full_dataset, subset_indices)
    log_message(f"[train_partial_kfold_new] Subset dataset size: {len(subset_dataset)}")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_mious = [-float('inf')] * k
    weights = torch.ones(num_classes, dtype=torch.float32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(subset_indices)):
        log_message(f"[train_partial_kfold_new] Starting Fold {fold+1}/{k}")

        train_subset = Subset(subset_dataset, train_idx)
        val_subset = Subset(subset_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = PointNet2Seg(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_batches = len(train_loader)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch_idx, (points, labels) in enumerate(train_loader):
                # Normalize points batch
                points = points - points.mean(dim=1, keepdim=True)
                norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
                points = points / norms
                points = points.permute(0, 2, 1).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(points)
                outputs = outputs.permute(0, 2, 1)

                loss = criterion(outputs.contiguous().view(-1, num_classes), labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    log_message(f"[train_partial_kfold_new] Fold {fold+1} Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / num_batches
            log_message(f"[train_partial_kfold_new] Fold {fold+1} Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

            val_loss, val_miou = validate(model, val_loader, criterion, device, num_classes)
            log_message(f"[train_partial_kfold_new] Fold {fold+1} Epoch {epoch+1} Validation Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")

            if val_miou > best_mious[fold]:
                best_mious[fold] = val_miou
                save_path = f"pointnet2_fold{fold+1}_epoch{epoch+1}_miou{val_miou:.4f}.pth"
                torch.save(model.state_dict(), save_path)
                log_message(f"[train_partial_kfold_new] Saved best model for fold {fold+1} at epoch {epoch+1} with mIoU {val_miou:.4f}")

    log_message("[train_partial_kfold_new] Training complete.")
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, samples=5):
            self.samples = samples
            self.points_num = points_num
        def __len__(self):
            return self.samples
        def __getitem__(self, idx):
            points = torch.rand(self.points_num, 3) * 2 - 1
            labels = torch.randint(0, num_classes, (self.points_num,))
            return points, labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PointNet2Seg(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for points, labels in loader:
            points = points - points.mean(dim=1, keepdim=True)
            norms, _ = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
            points = points / norms
            points = points.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs.contiguous().view(-1, num_classes), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Synthetic Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    print("[train_synthetic] Synthetic data training complete.")

def train_partial_kfold(sample_size=1000, batch_size=8, num_epochs=10, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    log_message("[train_partial_kfold] Loading full dataset...")
    full_dataset = CombinedLille1ParisDataset(data_dirs=["preprocessed_data"])
    log_message(f"[train_partial_kfold] Full dataset size: {len(full_dataset)}")

    sample_size = min(sample_size, len(full_dataset))
    subset_indices = list(range(sample_size))
    subset_dataset = Subset(full_dataset, subset_indices)
    log_message(f"[train_partial_kfold] Subset dataset size: {len(subset_dataset)}")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    weights = torch.ones(num_classes, dtype=torch.float32)
    best_mious = [-float('inf')] * k

    for fold, (train_idx, val_idx) in enumerate(kf.split(subset_indices)):
        log_message(f"[train_partial_kfold] Starting Fold {fold+1}/{k}")

        train_subset = Subset(subset_dataset, train_idx)
        val_subset = Subset(subset_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

        model = PointNet2Seg(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_batches = len(train_loader)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch_idx, (points, labels) in enumerate(train_loader):
                balanced_points, balanced_labels = prepare_balanced_batch(points, labels, jitter_std=0.01)
                balanced_points = balanced_points - balanced_points.mean(dim=1, keepdim=True)
                norms = balanced_points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
                balanced_points = balanced_points / norms
                balanced_points = balanced_points.permute(0, 2, 1).to(device)
                balanced_labels = balanced_labels.to(device)

                optimizer.zero_grad()
                outputs = model(balanced_points)
                outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs.contiguous().view(-1, num_classes), balanced_labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    log_message(f"[train_partial_kfold] Fold {fold+1} Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / num_batches
            log_message(f"[train_partial_kfold] Fold {fold+1} Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

            val_loss, val_miou = validate(model, val_loader, criterion, device, num_classes)
            log_message(f"[train_partial_kfold] Fold {fold+1} Epoch {epoch+1} Validation Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")

            if val_miou > best_mious[fold]:
                best_mious[fold] = val_miou
                save_path = f"pointnet2_fold{fold+1}_epoch{epoch+1}_miou{val_miou:.4f}.pth"
                torch.save(model.state_dict(), save_path)
                log_message(f"[train_partial_kfold] Saved best model for fold {fold+1} at epoch {epoch+1} with mIoU {val_miou:.4f}")

    log_message("[train_partial_kfold] Training complete.")

def train_kfold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    data_dir = "preprocessed_data"
    k = 5
    folds = get_folds(data_dir, k=k)
    
    weights = torch.ones(num_classes, dtype=torch.float32)

    for fold, (train_files, val_files) in enumerate(folds):
        log_message(f"Starting fold {fold+1}/{k}")

        train_dataset = CombinedLille1ParisDataset(files_lists=[train_files])
        val_dataset = CombinedLille1ParisDataset(files_lists=[val_files])

        log_message("Train dataset loaded, samples: " + str(len(train_dataset)))
        log_message("Validation dataset loaded, samples: " + str(len(val_dataset)))

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

        model = PointNet2Seg(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)

        num_epochs = 10
        best_miou = -float('inf')

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            # for batch_idx, (points, labels) in enumerate(train_loader):
            #     if batch_idx % 10 == 0:
            #         log_message(f"Fold {fold+1} Epoch {epoch+1} processing batch {batch_idx}")

            for points, labels in train_loader:
                balanced_points, balanced_labels = prepare_balanced_batch(points, labels, jitter_std=0.01)

                balanced_points = balanced_points - balanced_points.mean(dim=1, keepdim=True)
                norms = balanced_points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
                balanced_points = balanced_points / norms
                balanced_points = balanced_points.permute(0, 2, 1).to(device)
                balanced_labels = balanced_labels.to(device)

                optimizer.zero_grad()
                outputs = model(balanced_points)
                outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs.contiguous().view(-1, num_classes), balanced_labels.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            log_message(f"Fold {fold+1} Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

            val_loss, val_miou = validate(model, val_loader, criterion, device, num_classes)

            # Save model checkpoint if improvement
            if val_miou > best_miou:
                best_miou = val_miou
                model_save_path = f"pointnet2_fold{fold+1}_epoch{epoch+1}_miou{val_miou:.4f}.pth"
                torch.save(model.state_dict(), model_save_path)
                log_message(f"Saved best model to {model_save_path} with mIoU {best_miou:.4f}")

if __name__ == "__main__":
    # train_debug()
    # train_partial()
    # train_partial_kfold_new()
    train_partial_kfold()
    # train_kfold()
