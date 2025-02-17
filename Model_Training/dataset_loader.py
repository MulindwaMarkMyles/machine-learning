import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import scipy.io as sio


affected_data_path = "../ADHD_part1/ADHD_part1/v8p.mat"
healthy_data_path = "../Control_part1/Control_part1/v44p.mat"

affected_data = sio.loadmat(affected_data_path)
healthy_data = sio.loadmat(healthy_data_path)


affected = affected_data["v8p"]
healthy = healthy_data["v44p"]


affected_data_dataframe = pd.DataFrame(affected)
healthy_data_dataframe = pd.DataFrame(healthy)


class EEGDataset(Dataset):
    def __init__(self, dataset, label):
        self.data = dataset
        self.features = self.data.values.astype(np.float32)  # All columns are features
        self.labels = np.full((len(self.data),), label, dtype=np.int64)  # Assign fixed label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


# Example usage
def get_dataloader(dataset, label, batch_size=32, shuffle=True):
    dataset = EEGDataset(dataset, label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


healthy_dataloader = get_dataloader(affected_data_dataframe, label=0)
affected_dataloader = get_dataloader(healthy_data_dataframe, label=1)

# Print the number of samples in each dataset
print(f"Healthy samples: {len(healthy_dataloader.dataset)}")
print(f"Affected samples: {len(affected_dataloader.dataset)}")
