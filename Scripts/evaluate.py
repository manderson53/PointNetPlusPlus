import torch
import torch.nn.functional as F
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from Lille2Dataset import Lille2Dataset
from pointnetplusplus import PointNet2Seg

def evaluate():
    # -------------------------
    # Settings
    # -------------------------
    checkpoint_path = "checkpoints/6_pointnet2.pth"
    batch_size = 8
    num_classes = 10
    output_dir = "eval_results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset & Loader
    # -------------------------
    val_dataset = Lille2Dataset(data_dirs=["validation_data"])
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------
    # Load Model
    # -------------------------
    model = PointNet2Seg(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # -------------------------
    # Confusion Matrix
    # -------------------------
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for points, labels in tqdm(val_loader, desc="Evaluating"):
            # Make sure points have shape [B, 3, 4096]
            points = points.permute(0, 2, 1).to(device)  # (B, 3, N)
            labels = labels.to(device)

            logits = model(points)
            preds = logits.max(dim=2)[1]  # (B, N)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                if t < num_classes:
                    confusion_matrix[t.long(), p.long()] += 1

    # -------------------------
    # Metrics
    # -------------------------
    tp = torch.diag(confusion_matrix)
    fp = confusion_matrix.sum(0) - tp
    fn = confusion_matrix.sum(1) - tp

    support = confusion_matrix.sum(1).cpu().numpy()
    intersection = tp
    union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - tp
    per_class_iou = (intersection / union.clamp(min=1e-6)).cpu().numpy()
    iou_avg = per_class_iou.mean()
    iou_vote = (tp.sum() / union.sum()).item()

    precision = (tp / (tp + fp).clamp(min=1e-6)).cpu().numpy()
    recall = (tp / (tp + fn).clamp(min=1e-6)).cpu().numpy()
    f1 = 2 * precision * recall / (precision + recall).clip(min=1e-6)

    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()

    precision_micro = (tp.sum() / (tp.sum() + fp.sum()).clamp(min=1e-6)).item()
    recall_micro = (tp.sum() / (tp.sum() + fn.sum()).clamp(min=1e-6)).item()
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-6))

    # -------------------------
    # Save to CSV
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"eval_{os.path.basename(checkpoint_path)}.csv")

    rows = []
    for i in range(num_classes):
        rows.append({
            "Class": i,
            "Support": int(support[i]),
            "IoU": per_class_iou[i],
            "Precision": precision[i],
            "Recall": recall[i],
            "F1": f1[i]
        })

    rows.append({"Class": "Mean IoU (avg)", "IoU": iou_avg})
    rows.append({"Class": "IoU (vote)", "IoU": iou_vote})
    rows.append({"Class": "Macro", "Precision": precision_macro, "Recall": recall_macro, "F1": f1_macro})
    rows.append({"Class": "Micro", "Precision": precision_micro, "Recall": recall_micro, "F1": f1_micro})

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # -------------------------
    # Print Results
    # -------------------------
    print("\nPer-class IoU / Precision / Recall / F1 / Support:")
    for i in range(num_classes):
        print(f"Class {i}: IoU={per_class_iou[i]:.4f}, "
            f"P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}, Support={int(support[i])}")

    print(f"\nMean IoU (avg): {iou_avg:.4f}")
    print(f"IoU (vote): {iou_vote:.4f}")
    print(f"Macro - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Micro - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"\n✅ Results saved to {csv_path}")

if __name__ == "__main__":
    evaluate()