import torch
from torch.utils.data import DataLoader
from Lille2Dataset import Lille2Dataset
from pointnetplusplus import PointNet2Seg
from chat_gpt_training_k_fold import validate, num_classes, device

def main():
    # ----------------- Config -----------------
    data_dir = "validation_data"   # same dir you used for training
    checkpoint_path = "checkpoints_kfold/fold1_best_miou.pth"
    batch_size = 32

    # ----------------- Load dataset -----------------
    test_dataset = Lille2Dataset(data_dirs=[data_dir])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  
    # 👆 set num_workers=0 for Windows safety

    # ----------------- Load model -----------------
    model = PointNet2Seg(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from {checkpoint_path}")

    # ----------------- Run evaluation -----------------
    val_ce_loss, val_p_loss, val_sh_loss, val_total_loss, val_acc, metrics = validate(model, test_loader)

    print("\n===== Test Results =====")
    print(f"Total Loss: {val_total_loss:.4f}")
    print(f"Accuracy:   {val_acc:.4f}")
    print(f"Mean IoU:   {metrics['mean_iou']:.4f}")

    for i, iou in enumerate(metrics["per_class_iou"]):
        print(f"Class {i} IoU: {iou:.4f}")

if __name__ == "__main__":
    main()
