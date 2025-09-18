import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from yanx27_sem_seg import get_model
from ParisLilleDataset import ParisLilleDataset
from collections import defaultdict, Counter

try:
    from config_local import EVAL_OUTPUT_DIR, EVAL_OUTPUT_DIR_PHYSICS, TRAINING_OUTPUT_DIR, TRAINING_OUTPUT_DIR_PHYSICS
except ImportError:
    raise RuntimeError("Missing config_local.py. Please create it with Label Weights and OUTPUT_DIR defined.")

# -----------------------------
# Metric computation
# -----------------------------
def compute_metrics(y_true, y_pred, num_classes=10):
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)

    val_acc = np.mean(y_true == y_pred)  # same as OA here
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    ious = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious[cls] = iou
    mean_iou = np.mean(list(ious.values()))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0
    )

    per_class = {}
    for cls in range(num_classes):
        per_class[cls] = {
            "iou": ious[cls],
            "precision": precisions[cls],
            "recall": recalls[cls],
            "f1": f1s[cls],
            "support": supports[cls]
        }

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

# -----------------------------
# Compute overall mIoU for avg and vote
# -----------------------------
def compute_overall_miou(avg_row, vote_row, num_classes=10):
    # avg
    avg_iou_values = [avg_row[f"IoU_Class{cls_id}"] for cls_id in range(num_classes)]
    avg_row["Overall_mIoU"] = np.mean(avg_iou_values)

    # vote
    vote_iou_values = [vote_row[f"IoU_Class{cls_id}"] for cls_id in range(num_classes)]
    vote_row["Overall_mIoU"] = np.mean(vote_iou_values)

    return avg_row, vote_row

# -----------------------------
# Evaluation with avg and vote
# -----------------------------
def evaluate_best_models():
    num_classes = 10
    batch_size = 96
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_folds = 5
    use_physics_loss = True
    output_dir = EVAL_OUTPUT_DIR_PHYSICS if use_physics_loss else EVAL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("Loading validation dataset...")
    dataset = ParisLilleDataset(data_dir='preprocessed_validation_data', transform=None, load_orig=False)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    num_points = len(dataset)
    print(f"Validation dataset loaded, length: {len(dataset)}")

    # Store predictions per fold
    all_fold_preds = []

    # Store per-fold metrics
    detailed_results = []

    # -----------------------------
    # Evaluate each fold
    # -----------------------------
    for fold_idx in range(num_folds):
        fold = fold_idx + 1
        print(f"\nEvaluating fold {fold}...")
        input_dir = TRAINING_OUTPUT_DIR_PHYSICS if use_physics_loss else TRAINING_OUTPUT_DIR
        model_path = os.path.join(input_dir, f"best_model_fold{fold}.pth")
        model = get_model(num_classes=num_classes, in_channel=0).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for points, labels, block_names, xyz_min, xyz_max in tqdm(val_loader, desc=f"Fold {fold}"):
                points, labels = points.to(device), labels.to(device)
                preds, _ = model(points)
                all_preds.append(preds.argmax(dim=2))
                all_labels.append(labels)

        all_preds_flat = torch.cat(all_preds).reshape(-1)
        all_labels_flat = torch.cat(all_labels).reshape(-1)

        all_fold_preds.append(all_preds_flat)  # for ensemble voting

        metrics = compute_metrics(all_labels_flat, all_preds_flat, num_classes=num_classes)

        detailed_row = {
            "Fold": fold,
            "Val_Acc": metrics['val_acc'],
            "Mean_IoU": metrics['mean_iou'],
            "Precision_Macro": metrics["precision_macro"],
            "Recall_Macro": metrics["recall_macro"],
            "F1_Macro": metrics["f1_macro"],
            "Precision_Micro": metrics["precision_micro"],
            "Recall_Micro": metrics["recall_micro"],
            "F1_Micro": metrics["f1_micro"],
        }

        # Add per-class metrics
        for cls_id, cls_metrics in metrics['per_class'].items():
            detailed_row[f"IoU_Class{cls_id}"] = cls_metrics['iou']
            detailed_row[f"Prec_Class{cls_id}"] = cls_metrics['precision']
            detailed_row[f"Rec_Class{cls_id}"] = cls_metrics['recall']
            detailed_row[f"F1_Class{cls_id}"] = cls_metrics['f1']
            detailed_row[f"Support_Class{cls_id}"] = cls_metrics['support']

        detailed_results.append(detailed_row)

    # Save per-fold metrics CSV
    csv_path = os.path.join(output_dir, "evaluation_per_fold.csv")
    pd.DataFrame(detailed_results).to_csv(csv_path, index=False)
    print(f"Per-fold metrics saved to {csv_path}")

    # -----------------------------
    # Compute avg and vote metrics
    # -----------------------------
    labels_flat = all_labels_flat.clone()  # same across folds

    # --- AVG metrics ---
    avg_metrics_per_class = defaultdict(list)
    for fold_preds in all_fold_preds:
        m = compute_metrics(labels_flat, fold_preds, num_classes=num_classes)
        for cls_id, cls_metrics in m['per_class'].items():
            avg_metrics_per_class[cls_id].append(cls_metrics)

    avg_row = {"Metric": "avg"}
    precision_macro_list, recall_macro_list, f1_macro_list = [], [], []
    precision_micro_list, recall_micro_list, f1_micro_list = [], [], []
    val_acc_list = []

    for cls_id in range(num_classes):
        cls_list = avg_metrics_per_class[cls_id]
        avg_row[f"IoU_Class{cls_id}"] = np.mean([c['iou'] for c in cls_list])
        avg_row[f"Prec_Class{cls_id}"] = np.mean([c['precision'] for c in cls_list])
        avg_row[f"Rec_Class{cls_id}"] = np.mean([c['recall'] for c in cls_list])
        avg_row[f"F1_Class{cls_id}"] = np.mean([c['f1'] for c in cls_list])
        avg_row[f"Support_Class{cls_id}"] = np.mean([c['support'] for c in cls_list])

    # Compute macro/micro/val_acc/OA for avg
    metrics_list = [compute_metrics(labels_flat, p, num_classes=num_classes) for p in all_fold_preds]
    avg_row["Precision_Macro"] = np.mean([m['precision_macro'] for m in metrics_list])
    avg_row["Recall_Macro"] = np.mean([m['recall_macro'] for m in metrics_list])
    avg_row["F1_Macro"] = np.mean([m['f1_macro'] for m in metrics_list])
    avg_row["Precision_Micro"] = np.mean([m['precision_micro'] for m in metrics_list])
    avg_row["Recall_Micro"] = np.mean([m['recall_micro'] for m in metrics_list])
    avg_row["F1_Micro"] = np.mean([m['f1_micro'] for m in metrics_list])
    avg_row["OA"] = np.mean([m['val_acc'] for m in metrics_list])

    # --- VOTE metrics (majority vote) ---
    preds_stack = torch.stack(all_fold_preds)  # [num_folds, num_points]
    vote_preds = torch.mode(preds_stack, dim=0)[0]

    vote_metrics = compute_metrics(labels_flat, vote_preds, num_classes=num_classes)
    vote_row = {"Metric": "vote"}
    for cls_id, cls_metrics in vote_metrics['per_class'].items():
        vote_row[f"IoU_Class{cls_id}"] = cls_metrics['iou']
        vote_row[f"Prec_Class{cls_id}"] = cls_metrics['precision']
        vote_row[f"Rec_Class{cls_id}"] = cls_metrics['recall']
        vote_row[f"F1_Class{cls_id}"] = cls_metrics['f1']
        vote_row[f"Support_Class{cls_id}"] = cls_metrics['support']

    vote_row["Precision_Macro"] = vote_metrics["precision_macro"]
    vote_row["Recall_Macro"] = vote_metrics["recall_macro"]
    vote_row["F1_Macro"] = vote_metrics["f1_macro"]
    vote_row["Precision_Micro"] = vote_metrics["precision_micro"]
    vote_row["Recall_Micro"] = vote_metrics["recall_micro"]
    vote_row["F1_Micro"] = vote_metrics["f1_micro"]
    vote_row["OA"] = vote_metrics["val_acc"]

    # Compute overall mIoU
    avg_row, vote_row = compute_overall_miou(avg_row, vote_row, num_classes=num_classes)

    # Save avg + vote metrics
    avg_vote_csv_path = os.path.join(output_dir, "evaluation_avg_vote.csv")
    pd.DataFrame([avg_row, vote_row]).to_csv(avg_vote_csv_path, index=False)
    print(f"Avg and vote metrics saved to {avg_vote_csv_path}")

if __name__ == "__main__":
    evaluate_best_models()