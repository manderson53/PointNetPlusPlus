from CombinedLille1ParisDataset import CombinedLille1ParisDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Set path to your preprocessed .npy folder
    data_dir = "preprocessed_data"

    dataset = CombinedLille1ParisDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # Quick test
    for i, (points, labels) in enumerate(dataloader):
        print("Batch", i)
        print("Points shape:", points.shape)   # [8, 4096, 3]
        print("Labels shape:", labels.shape)   # [8, 4096]
        break
