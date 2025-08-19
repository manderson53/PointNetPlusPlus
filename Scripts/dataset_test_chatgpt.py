import torch
from torch.utils.data import DataLoader, random_split
from CombinedLille1ParisDataset import CombinedLille1ParisDataset


def main():
    # Params
    batch_size = 8
    val_split = 0.2
    data_dir = "preprocessed_data"

    # Load dataset
    dataset = CombinedLille1ParisDataset(data_dirs=[data_dir])

    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Debug: inspect one batch
    print("==== Checking train loader ====")
    for points, labels in train_loader:
        print("Points shape:", points.shape)   # expect [B, N, 3]
        print("Labels shape:", labels.shape)   # expect [B, N]
        print("Points dtype:", points.dtype)
        print("Labels dtype:", labels.dtype)
        break

    print("\n==== Checking val loader ====")
    for points, labels in val_loader:
        print("Val Points shape:", points.shape)
        print("Val Labels shape:", labels.shape)
        print("Val Points dtype:", points.dtype)
        print("Val Labels dtype:", labels.dtype)
        break

if __name__ == "__main__":
    main()  
