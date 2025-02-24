import os
import numpy  as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.decomposition import PCA
import torch_xla
import torch_xla.core.xla_model as xm  # XLA Model for TPU
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu

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

class DataAugmentation:
    def __init__(self, noise_level=0.05, dropout_prob=0.1):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob

    def __call__(self, data):
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, data.shape)
        data = data + noise

        # Random dropout
        mask = np.random.binomial(1, 1-self.dropout_prob, data.shape)
        data = data * mask

        return data


class BaseNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(BaseNetwork, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

class LinearRegressionNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(LinearRegressionNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class LogisticRegressionNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(LogisticRegressionNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class TreeBasedNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(TreeBasedNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class SVMNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(SVMNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class KNNNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(KNNNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class GBMNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(GBMNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class XGBoostNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(XGBoostNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class DeepNeuralNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(DeepNeuralNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class PCANet(BaseNetwork):
    def __init__(self, input_dim, embed_dim, n_components=None):
        super(PCANet, self).__init__(input_dim, embed_dim)
        if n_components is None:
            n_components = min(input_dim, embed_dim)
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        
        self.features = nn.Sequential(
            nn.Linear(n_components, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # Convert to numpy for PCA
        x_numpy = x.cpu().detach().numpy()
        
        # Fit PCA only once with the first batch
        if not self.fitted:
            self.pca.fit(x_numpy)
            self.fitted = True
        
        # Transform the data using PCA
        x_pca = self.pca.transform(x_numpy)
        
        # Convert back to tensor
        x_pca = torch.FloatTensor(x_pca).to(x.device)
        
        return self.features(x_pca)

def contrastive_loss(student_output, teacher_output, label, margin=1.0):
    distance = torch.norm(student_output - teacher_output, p=2)
    loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

def train_epoch(model, teacher_model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrap DataLoader with ParallelLoader for TPU
    para_loader = pl.MpDeviceLoader(dataloader, device)

    for data, labels in para_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        teacher_outputs = teacher_model(data)

        ce_loss = criterion(outputs.squeeze(), labels.float())
        kd_loss = nn.MSELoss()(outputs, teacher_outputs.detach())
        cont_loss = contrastive_loss(outputs, teacher_outputs.detach(), labels)

        loss = 0.5 * ce_loss + 0.3 * kd_loss + 0.2 * cont_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Use TPU optimizer step
        xm.optimizer_step(optimizer)

        running_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).int()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs.squeeze(), label.float())
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).int()
            correct += (predicted == label).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy


def train_model(model_name, model_class, dataset, device, teacher_embed_dim=1024, student_embed_dim=512):

    os.makedirs("/kaggle/working/models/", exist_ok=True)
    
    # Create dataloaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Initialize models
    input_dim = dataset[0][0].shape[1] * dataset[0][0].shape[0]
    
    # Initialize teacher and student models
    teacher_model = model_class(input_dim, teacher_embed_dim).to(device)
    student_model = model_class(input_dim, student_embed_dim).to(device)

    # Save model configuration
    model_config = {
        'input_dim': input_dim,
        'embed_dim': student_embed_dim,
        'max_rows': dataset.max_rows,
        'num_features': dataset[0][0].shape[1]
    }
    joblib.dump(model_config, f"/kaggle/working/models/{model_name}_config.pkl")

    optimizer = optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCELoss()

    # Training loop
    num_epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # ... Keep existing train_epoch and validate functions ...
        train_loss, train_acc = train_epoch(student_model, teacher_model, train_loader, optimizer, device, criterion)
        val_loss, val_acc = validate(student_model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), f"/kaggle/working/models/{model_name}_best.pth")
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save final model and scaler
    torch.save(student_model.state_dict(), f"/kaggle/working/models/{model_name}_final.pth")
    joblib.dump(dataset.scaler, f"/kaggle/working/models/{model_name}_scaler.pkl")
    
    return best_val_acc

if __name__ == "__main__":
    # Setup paths and dataset
    adhd_folder = "/kaggle/input/adhd-dataset/ADHD_part2/ADHD_part2"
    control_folder = "/kaggle/input/adhd-dataset/Control_part2/Control_part2"
    
    transform = DataAugmentation(noise_level=0.05, dropout_prob=0.1)
    dataset = MATDataset(adhd_folder, control_folder, transform=transform)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = xm.xla_device()
    
    # Dictionary of models to train
    models = {
        'linear': LinearRegressionNet,
        'logistic': LogisticRegressionNet,
        'tree': TreeBasedNet,
        'forest': TreeBasedNet,  # Using same architecture with different initialization
        'svm': SVMNet,
        'knn': KNNNet,
        'gbm': GBMNet,
        'xgboost': XGBoostNet,
        'neural': DeepNeuralNet,
        'pca': PCANet  # Add PCA-based model
    }

    # Train all models
    results = {}
    for model_name, model_class in models.items():
        print(f"\nTraining {model_name} model...")
        best_acc = train_model(model_name, model_class, dataset, device)
        results[model_name] = best_acc
        print(f"Completed training {model_name} model. Best accuracy: {best_acc:.4f}")

    # Print final results
    print("\nFinal Results:")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.4f}")
