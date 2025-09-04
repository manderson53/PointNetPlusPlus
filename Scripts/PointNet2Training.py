import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from yanx27_sem_seg import get_model
from ParisLilleDataset import ParisLilleDataset


# --- Global Variables ---
ground_points = {
    'Lille1_1': np.array([
        [-370.49194, -419.67587, 37.003822],
        [  98.32042, -419.67587, 37.003822],
        [-370.49194,  115.57095, 44.557846],
        [  98.32042,  115.57095, 44.557846]
    ]),
    'Lille1_2': np.array([
        [-777.96643, -761.9154, 25.609138],
        [-318.567,   -761.9154, 25.609138],
        [-777.96643, -362.2454, 34.769135],
        [-318.567,   -362.2454, 34.769135]
    ]),
    'Paris': np.array([
        [  3.1266403, -781.76514, 42.56062],
        [ 74.77652,   -781.76514, 42.56062],
        [  3.1266403, -290.8174,  35.06052],
        [ 74.77652,   -290.8174,  35.06052]
    ])
}

def physics_loss(points_raw, logits, block_names, xyz_min, xyz_max, bld_plane_margin=0.1, bld_below_weight=0.2):
    """
    Compute physics-based penalties for semantic segmentation.

    Args:
        points_raw (Tensor): normalized input points [B, N, 3] in [0,1]
        logits      (Tensor): model output logits [B, N, C] or [B, C, N]
        block_names (list[str]): block name per batch element
        xyz_min, xyz_max (list[Tensor]): per-block min/max coordinates [B,3]
        bld_plane_margin (float): min height above plane for buildings
        bld_below_weight (float): weight for building-under-plane penalty

    Returns:
        Tensor: scalar physics loss
    """
    device = points_raw.device
    B, N = points_raw.shape[:2]

    # Stack per-block min/max for the batch
    xyz_min_batch = torch.stack(xyz_min).to(device)  # [B,3]
    xyz_max_batch = torch.stack(xyz_max).to(device)  # [B,3]

    # De-normalize each point cloud individually
    scale = xyz_max_batch - xyz_min_batch           # [B,3]
    P = points_raw * scale.unsqueeze(2) + xyz_min_batch.unsqueeze(2)  # [B,3,N]
    X, Y, Z = P[:,0,:], P[:,1,:], P[:,2,:]        # [B,N]

    # Standardize logits to [B, N, C]
    if logits.dim() == 3 and logits.shape[1] != N:
        logits = logits.permute(0, 2, 1).contiguous()
    probs = torch.softmax(logits, dim=2)  # [B, N, C]

    # Class probabilities
    g   = probs[..., 1]    # ground
    bld = probs[..., 2]    # building
    ped = probs[..., 7]    # pedestrian
    car = probs[..., 8]    # car
    trash = probs[..., 5]  # trash can
    boll = probs[..., 4]   # bollard / small pole

    # Compute per-sample plane coefficients
    a = torch.zeros(B, 1, device=device, dtype=P.dtype)
    b = torch.zeros_like(a)
    c = torch.zeros_like(a)

    for bi, name in enumerate(block_names):
        if name in ground_points:
            pts = ground_points[name]
            aa, bb, cc = compute_plane_coeff(pts)
            a[bi, 0], b[bi, 0], c[bi, 0] = aa, bb, cc
        else:
            # fallback: use 5th percentile of Z for the block
            c[bi, 0] = torch.quantile(Z[bi], 0.05)

    # Height above ground
    z_ground = a * X + b * Y + c   # [B, N]
    dz = Z - z_ground
    hag = dz.clamp_min(0.0)

    # Penalty thresholds (meters)
    loss_ground     = (g   * F.relu(hag - 0.30)).mean()
    loss_ped        = (ped * F.relu(hag - 2.20)).mean()
    loss_car        = (car * F.relu(hag - 3.00)).mean()
    loss_trash      = (trash * F.relu(hag - 2.00)).mean()
    loss_boll       = (boll * F.relu(hag - 1.50)).mean()
    loss_bld_above  = (bld * F.relu(bld_plane_margin - dz)).mean()

    total_phys_loss = loss_ground + loss_ped + loss_car + loss_trash + loss_boll + bld_below_weight * loss_bld_above
    return total_phys_loss

def compute_plane_coeff(points):
    """
    points: np.array of shape [4,3] (x,y,z)
    Returns: a,b,c such that z = a*x + b*y + c
    """
    p1, p2, p3 = points[:3]
    # Solve linear system
    A = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])
    Z = np.array([p1[2], p2[2], p3[2]])
    a, b, c = np.linalg.solve(A, Z)
    return float(a), float(b), float(c)

def compute_metrics(y_true, y_pred, num_classes=10):

    
    """
    Compute metrics for semantic segmentation / classification.
    
    Args:
        y_true (Tensor): Ground truth labels (N,)
        y_pred (Tensor): Predicted labels (N,)
        num_classes (int): Number of classes
        
    Returns:
        dict with val_acc, mean_iou, per_class metrics, macro/micro P/R/F1
    """
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)

    # Accuracy
    val_acc = np.mean(y_true == y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # IoU per class
    ious = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious[cls] = iou
    mean_iou = np.mean(list(ious.values()))

    # Precision, Recall, F1, Support per class
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0
    )

    # Per-class dict
    per_class = {}
    for cls in range(num_classes):
        per_class[cls] = {
            "iou": ious[cls],
            "precision": precisions[cls],
            "recall": recalls[cls],
            "f1": f1s[cls],
            "support": supports[cls]
        }

    # Macro & Micro
    precision_macro = precisions.mean()
    recall_macro = recalls.mean()
    f1_macro = f1s.mean()

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    return {
        "val_acc": val_acc,
        "mean_iou": mean_iou,
        "per_class": per_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }

def train_kfold():
    # --------------------------
    num_folds = 5
    num_epochs = 50
    batch_size = 96
    num_classes = 10
    learning_rate = 0.001
    use_physics_loss = False
    output_dir = "training_results"
    os.makedirs(output_dir, exist_ok=True)

    label_weights_path = r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Scripts\preprocessed_training_data\labelweights.npy"
    label_weights = np.load(label_weights_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_weights_tensor = torch.tensor(label_weights, dtype=torch.float32).to(device)
    print(f"[{time.strftime('%H:%M:%S')}] Loading Dataset...")
    dataset = ParisLilleDataset(data_dir='preprocessed_training_data', transform=None, load_orig=False)
    print(f"[{time.strftime('%H:%M:%S')}] Dataset loaded, length:", len(dataset))

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    summary_path = os.path.join(output_dir, "summary.csv")
    detailed_path = os.path.join(output_dir, "detailed_metrics.csv")
    summary_data, detailed_data = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Training fold {fold} =====")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        model = get_model(num_classes=num_classes, in_channel=0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss(weight=label_weights_tensor)
        best_miou = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"[{time.strftime('%H:%M:%S')}] Starting epoch {epoch}")
            model.train()
            train_loss_ce_total, train_loss_phys_total, train_loss_total = 0.0, 0.0, 0.0

            for points, labels, block_names, xyz_min, xyz_max in tqdm(train_loader, desc=f"Epoch {epoch} "):
                points, labels = points.to(device), labels.to(device)
                optimizer.zero_grad()
                preds, _ = model(points)

                loss_ce = criterion(preds.reshape(-1, num_classes), labels.reshape(-1))
                if use_physics_loss:
                    loss_phys = physics_loss(points_raw=points, logits=preds, block_names=block_names,
                                             xyz_min=xyz_min, xyz_max=xyz_max)
                else:
                    loss_phys = 0.0

                lambda_phys = 0.1
                loss_total = loss_ce + lambda_phys * (loss_phys if isinstance(loss_phys, torch.Tensor) else 0.0)

                loss_total.backward()
                optimizer.step()

                train_loss_ce_total += loss_ce.item()
                train_loss_phys_total += loss_phys.item() if isinstance(loss_phys, torch.Tensor) else loss_phys
                train_loss_total += loss_total.item()

            # Average training losses
            train_loss_ce_total /= len(train_loader)
            train_loss_phys_total /= len(train_loader)
            train_loss_total /= len(train_loader)

            print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch} finished training, starting validation...")

            # Validation
            model.eval()
            val_loss_ce_total, val_loss_phys_total, val_loss_total = 0.0, 0.0, 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for points, labels, block_names, xyz_min, xyz_max in val_loader:
                    points, labels = points.to(device), labels.to(device)
                    preds, _ = model(points)

                    loss_ce = criterion(preds.reshape(-1, num_classes), labels.reshape(-1))
                    if use_physics_loss:
                        loss_phys = physics_loss(points_raw=points, logits=preds, block_names=block_names,
                                                 xyz_min=xyz_min, xyz_max=xyz_max)
                    else:
                        loss_phys = 0.0

                    loss_total = loss_ce + lambda_phys * (loss_phys if isinstance(loss_phys, torch.Tensor) else 0.0)
                    val_loss_ce_total += loss_ce.item()
                    val_loss_phys_total += loss_phys.item() if isinstance(loss_phys, torch.Tensor) else loss_phys
                    val_loss_total += loss_total.item()

                    all_preds.append(preds.argmax(dim=2))
                    all_labels.append(labels)

            val_loss_ce_total /= len(val_loader)
            val_loss_phys_total /= len(val_loader)
            val_loss_total /= len(val_loader)

            print(f"[{time.strftime('%H:%M:%S')}] Validation finished for epoch {epoch}. Computing metrics...")

            val_metrics = compute_metrics(torch.cat(all_labels).reshape(-1), torch.cat(all_preds).reshape(-1))
            mean_iou = val_metrics['mean_iou']
            is_best = mean_iou > best_miou
            if is_best:
                best_miou = mean_iou
                torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_fold{fold}.pth'))

            if epoch % 20 == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, f'model_fold{fold}_epoch{epoch}.pth'))

            print(f"[{time.strftime('%H:%M:%S')}] " 
                f"[Epoch {epoch}] "
                f"Train Loss={train_loss_total:.4f} "
                f"Val Loss={val_loss_total:.4f} "
                f"Val Acc={val_metrics['val_acc']:.4f} "
                f"mIoU={mean_iou:.4f}")

            # --- Save metrics ---
            summary_row = [fold, epoch, optimizer.param_groups[0]['lr'], train_loss_total, val_loss_total,
                           val_metrics['val_acc'], mean_iou, int(is_best)]
            for cls_id in sorted(val_metrics['per_class'].keys()):
                summary_row.append(val_metrics['per_class'][cls_id]['iou'])
            summary_data.append(summary_row)

            for cls_id, cls_metrics in val_metrics['per_class'].items():
                detailed_data.append([
                    fold, epoch, cls_id,
                    cls_metrics['iou'], cls_metrics['precision'], cls_metrics['recall'], cls_metrics['f1'], cls_metrics['support'],
                    val_loss_ce_total, val_loss_phys_total, val_loss_total,
                    train_loss_ce_total, train_loss_phys_total, train_loss_total
                ])

            # Macro / Micro
            for key, metrics in [('Macro', ('precision_macro', 'recall_macro', 'f1_macro')),
                                 ('Micro', ('precision_micro', 'recall_micro', 'f1_micro'))]:
                detailed_data.append([
                    fold, epoch, key,
                    None, val_metrics[metrics[0]], val_metrics[metrics[1]], val_metrics[metrics[2]], None,
                    val_loss_ce_total, val_loss_phys_total, val_loss_total,
                    train_loss_ce_total, train_loss_phys_total, train_loss_total
                ])

            scheduler.step()

            # Write CSVs
            summary_columns = ['Fold','Epoch','LR','Train_Loss','Val_Loss','Val_Acc','Mean_IoU','Best_mIoU_Flag'] \
                              + [f'IoU_Class{cls}' for cls in range(num_classes)]
            pd.DataFrame(summary_data, columns=summary_columns).to_csv(summary_path, index=False)

            detailed_columns = [
                'Fold','Epoch','Class','IoU','Precision','Recall','F1','Support',
                'Val_CE_Loss','Val_Physics_Loss','Val_Total_Loss',
                'Train_CE_Loss','Train_Physics_Loss','Train_Total_Loss'
            ]
            pd.DataFrame(detailed_data, columns=detailed_columns).to_csv(detailed_path, index=False)

    print("\nTraining complete! Summary and detailed metrics saved.")

if __name__ == "__main__":
    train_kfold()