import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

class MATDataset(Dataset):
    def __init__(self, adhd_folder, control_folder, max_rows=None, transform=None):
        self.adhd_files = [os.path.join(adhd_folder, f) for f in os.listdir(adhd_folder) if f.endswith('.mat')]
        self.control_files = [os.path.join(control_folder, f) for f in os.listdir(control_folder) if f.endswith('.mat')]
        self.files = self.adhd_files + self.control_files
        self.labels = [1] * len(self.adhd_files) + [0] * len(self.control_files)  # 1 for ADHD, 0 for Control

        # Determine the maximum number of rows across all files
        self.max_rows = max_rows
        if self.max_rows is None:
            self.max_rows = 0
            for file_path in self.files:
                data_dict = loadmat(file_path)
                for key in data_dict:
                    if not key.startswith('__'):
                        data = data_dict[key]
                        if isinstance(data, np.ndarray):
                            self.max_rows = max(self.max_rows, data.shape[0])

        # Use RobustScaler instead of StandardScaler for better handling of outliers
        self.scaler = RobustScaler()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        data_dict = loadmat(file_path)  # Load .mat file

        # Find the key with the largest number of values
        max_key = None
        max_size = 0
        for key in data_dict:
            if not key.startswith('__'):  # Ignore metadata keys
                data = data_dict[key]
                if isinstance(data, np.ndarray) and data.size > max_size:
                    max_key = key
                    max_size = data.size

        if max_key is None:
            raise ValueError(f"No valid data found in file: {file_path}")

        data = data_dict[max_key]  # Use the key with the largest data

        # Pad or truncate the data to have the same number of rows
        if data.shape[0] < self.max_rows:
            # Pad with zeros
            padding = np.zeros((self.max_rows - data.shape[0], data.shape[1]))
            data = np.vstack((data, padding))
        elif data.shape[0] > self.max_rows:
            # Truncate
            data = data[:self.max_rows, :]

        # Apply normalization per feature
        if not hasattr(self, 'fitted_scaler'):
            self.scaler.fit(data)
            self.fitted_scaler = True
        data = self.scaler.transform(data)

        # Apply data augmentation if specified
        if self.transform:
            data = self.transform(data)

        data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor
        return data, label


def load_ensemble_model(model_path):
    checkpoint = torch.load(model_path)
    return checkpoint['ensemble'], checkpoint['model_names'], checkpoint['metrics']

def get_dataset():
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    return MATDataset(adhd_folder, control_folder)

def get_feature_names(dataset):
    # Replace with actual feature names if available
    return [f"feature_{i}" for i in range(dataset[0][0].numel())]

def prepare_data(dataset):
    X = []
    y = []
    for data, label in dataset:
        X.append(data.numpy().flatten())
        y.append(label)  # Remove .numpy() since label is already an integer
    return np.array(X), np.array(y)
