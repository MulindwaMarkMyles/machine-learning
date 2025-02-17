import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

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

class EEGNet(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(EEGNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x

def contrastive_loss(student_output, teacher_output, label, margin=1.0):
    distance = torch.norm(student_output - teacher_output, p=2)
    loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

def train_epoch(model, teacher_model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(data)
        teacher_outputs = teacher_model(data)
        
        # Combine BCE loss and knowledge distillation
        ce_loss = criterion(outputs.squeeze(), labels.float())
        kd_loss = nn.MSELoss()(outputs, teacher_outputs.detach())
        loss = 0.7 * ce_loss + 0.3 * kd_loss
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
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

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define paths
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"

    # Create dataset with augmentation
    transform = DataAugmentation(noise_level=0.05, dropout_prob=0.1)
    dataset = MATDataset(adhd_folder, control_folder, transform=transform)

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize models
    input_dim = dataset[0][0].shape[1] * dataset[0][0].shape[0]
    print(f"Input dimension: {input_dim}")  # Debug print
    
    # Save input dimension for inference
    model_config = {
        'input_dim': input_dim,
        'embed_dim': 128,
        'max_rows': dataset.max_rows,
        'num_features': dataset[0][0].shape[1]
    }
    joblib.dump(model_config, "./models/model_config.pkl")
    
    teacher_model = EEGNet(input_dim, embed_dim=256)
    student_model = EEGNet(input_dim, embed_dim=128)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCELoss()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    # Training loop
    num_epochs = 100
    best_val_acc = 0.0  # Track best validation accuracy
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(student_model, teacher_model, train_loader, optimizer, device, criterion)
        
        # Validation phase
        val_loss, val_acc = validate(student_model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save model if it achieves better validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), "./models/best_model_three.pth")
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save final model and scaler
    torch.save(student_model.state_dict(), "./models/final_model_three.pth")
    joblib.dump(dataset.scaler, "./models/model_three_scaler.pkl")
    print(f"Training complete. Final model saved. Best validation accuracy was: {best_val_acc:.4f}")