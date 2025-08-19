from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from pointnetplusplus import PointNet2Seg
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import csv

# ----------------- Config -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
num_epochs = 100   # reduce if runtime is too long
num_folds = 5
data_dir = "preprocessed_data"
save_dir = "checkpoints_kfold"
os.makedirs(save_dir, exist_ok=True)

# ----------------- Class Weights -----------------
counts = torch.tensor([1462643, 53034315, 25824813, 546275, 54421,
                       180665, 3997055, 160917, 3442269, 9682547], dtype=torch.float32)
weights = 1.0 / counts
weights = weights / weights.sum() * len(counts)
weights = weights.to(device)

# ----------------- Metrics Helper -----------------
def compute_metrics(all_preds, all_labels, num_classes):
    metrics = {}
    ious, precision, recall, f1, support = [], [], [], [], []

    for cls in range(num_classes):
        tp = ((all_preds == cls) & (all_labels == cls)).sum().item()
        fp = ((all_preds == cls) & (all_labels != cls)).sum().item()
        fn = ((all_preds != cls) & (all_labels == cls)).sum().item()

        supp = (all_labels == cls).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_s = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        ious.append(iou)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_s)
        support.append(supp)

    metrics["per_class_iou"] = ious
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    metrics["support"] = support
    metrics["mean_iou"] = np.mean(ious)
    metrics["precision_macro"] = np.mean(precision)
    metrics["recall_macro"] = np.mean(recall)
    metrics["f1_macro"] = np.mean(f1)

    # Micro
    total_tp = sum(((all_preds == i) & (all_labels == i)).sum().item() for i in range(num_classes))
    total_fp = sum(((all_preds == i) & (all_labels != i)).sum().item() for i in range(num_classes))
    total_fn = sum(((all_preds != i) & (all_labels == i)).sum().item() for i in range(num_classes))
    prec_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_micro = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) > 0 else 0

    metrics["precision_micro"] = prec_micro
    metrics["recall_micro"] = rec_micro
    metrics["f1_micro"] = f1_micro

    return metrics

# ----------------- Validation Function -----------------
def validate(model, val_loader):
    model.eval()
    running_ce_loss = 0.0
    running_p_loss = 0.0
    running_sh_loss = 0.0
    correct = 0
    total_points = 0
    criterion = nn.CrossEntropyLoss(weight=weights)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for points, labels in val_loader:
            # Normalize points
            points = points - points.mean(dim=1, keepdim=True)
            norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
            points = points / norms
            points = points.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(points)

            # Compute individual losses
            ce_loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            p_loss = torch.tensor(0.0, device=device)   # placeholder
            sh_loss = torch.tensor(0.0, device=device)  # placeholder

            # Accumulate losses weighted by batch size
            num_points = labels.numel()
            running_ce_loss += ce_loss.item() * num_points
            running_p_loss += p_loss.item() * num_points
            running_sh_loss += sh_loss.item() * num_points
            total_points += num_points

            # Predictions for metrics
            preds = outputs.permute(0, 2, 1).contiguous().view(-1, num_classes).argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.view(-1).cpu())

            correct += (preds == labels.view(-1)).sum().item()

    # Average losses over all points
    val_ce_loss = running_ce_loss / total_points
    val_p_loss = running_p_loss / total_points
    val_sh_loss = running_sh_loss / total_points
    val_total_loss = val_ce_loss + val_p_loss + val_sh_loss
    val_acc = correct / total_points

    # Compute per-class metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels, num_classes)

    return val_ce_loss, val_p_loss, val_sh_loss, val_total_loss, val_acc, metrics

# ----------------- K-Fold Training -----------------
def train_kfold():
    dataset = CombinedLille1ParisDataset(data_dirs=[data_dir])
    print("Full dataset size:", len(dataset))

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    best_val_loss = float("inf")
    best_miou = -1.0

    # CSV logging
    csv_file = os.path.join(save_dir, "kfold_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Fold", "Epoch", "Train_CE_Loss", "Train_P_Loss", "Train_Sh_Loss", "Train_Total_Loss",
                  "Val_CE_Loss", "Val_P_Loss", "Val_Sh_Loss", "Val_Total_Loss", "Val_Acc", "Mean_IoU"]
        for c in range(num_classes):
            header += [f"IoU_Class{c}", f"P_Class{c}", f"R_Class{c}", f"F1_Class{c}", f"Support_Class{c}"]
        header += ["Precision_Macro", "Recall_Macro", "F1_Macro", "Precision_Micro", "Recall_Micro", "F1_Micro"]
        writer.writerow(header)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold+1}/{num_folds} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

        model = PointNet2Seg(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(weight=weights)

        for epoch in range(num_epochs):
            model.train()
            running_ce_loss, running_p_loss, running_sh_loss = 0, 0, 0
            total_points = 0

            for points, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}"):
                points = points - points.mean(dim=1, keepdim=True)
                norms = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
                points = points / norms
                points = points.permute(0, 2, 1).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(points)

                ce_loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
                p_loss = torch.tensor(0.0, device=device)
                sh_loss = torch.tensor(0.0, device=device)
                total_loss = ce_loss + p_loss + sh_loss

                total_loss.backward()
                optimizer.step()

                num_points = labels.numel()   # number of points in this batch
                running_ce_loss += ce_loss.item() * num_points
                running_p_loss += p_loss.item() * num_points
                running_sh_loss += sh_loss.item() * num_points
                total_points += num_points

            train_ce_loss = running_ce_loss / total_points
            train_p_loss = running_p_loss / total_points
            train_sh_loss = running_sh_loss / total_points
            train_total_loss = train_ce_loss + train_p_loss + train_sh_loss

            val_ce_loss, val_p_loss, val_sh_loss, val_total_loss, val_acc, metrics = validate(model, val_loader)

            print(f"Fold {fold+1} Epoch {epoch+1} | Train Total Loss: {train_total_loss:.4f} | "
                  f"Val Total Loss: {val_total_loss:.4f} | Val Acc: {val_acc:.4f} | mIoU: {metrics['mean_iou']:.4f}")

            # Save metrics
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                row = [fold+1, epoch+1, train_ce_loss, train_p_loss, train_sh_loss, train_total_loss,
                       val_ce_loss, val_p_loss, val_sh_loss, val_total_loss, val_acc, metrics["mean_iou"]]
                for i in range(num_classes):
                    row += [metrics["per_class_iou"][i], metrics["precision"][i],
                            metrics["recall"][i], metrics["f1"][i], metrics["support"][i]]
                row += [metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"],
                        metrics["precision_micro"], metrics["recall_micro"], metrics["f1_micro"]]
                writer.writerow(row)

            # Save best / snapshots
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold+1}_best_val_loss.pth"))

            if metrics["mean_iou"] > best_miou:
                best_miou = metrics["mean_iou"]
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold+1}_best_miou.pth"))
            
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold+1}_epoch{epoch+1}.pth"))

        fold_results.append(best_val_loss)

    print("\n===== Cross-validation results =====")
    print(f"Mean Val Loss: {np.mean(fold_results):.4f}, Std: {np.std(fold_results):.4f}")


if __name__ == "__main__":
    train_kfold()
