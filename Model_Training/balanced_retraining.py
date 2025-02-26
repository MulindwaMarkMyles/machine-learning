import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import joblib

# Import model classes (assuming they're defined in multimodel_train.py)
from multimodel_train import (LinearRegressionNet, TreeBasedNet, GBMNet)

class BalancedMATDataset(Dataset):
    """Dataset class with improved preprocessing and balance monitoring"""
    def __init__(self, adhd_folder, control_folder, transform=None, normalize=True):
        self.transform = transform
        self.adhd_files = [os.path.join(adhd_folder, f) for f in os.listdir(adhd_folder) if f.endswith('.mat')]
        self.control_files = [os.path.join(control_folder, f) for f in os.listdir(control_folder) if f.endswith('.mat')]
        
        print(f"Found {len(self.adhd_files)} ADHD files and {len(self.control_files)} control files")
        
        # Balance dataset by taking the minimum number of files from each class
        min_files = min(len(self.adhd_files), len(self.control_files))
        self.adhd_files = self.adhd_files[:min_files]
        self.control_files = self.control_files[:min_files]
        
        print(f"Using {len(self.adhd_files)} files from each class for balance")
        
        # Combine all files with labels
        self.files = [(f, 1) for f in self.adhd_files] + [(f, 0) for f in self.control_files]
        
        # Determine max dimensions and which key to use in each file
        print("Determining dataset dimensions...")
        self.file_keys = {}
        self.max_rows = 0
        self.n_features = 0
        
        for file_path, _ in tqdm(self.files):
            data_dict = loadmat(file_path)
            
            # Find the key with the largest data
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
                
            self.file_keys[file_path] = max_key
            data = data_dict[max_key]
            
            # Update dimensions
            self.max_rows = max(self.max_rows, data.shape[0])
            if self.n_features == 0:
                self.n_features = data.shape[1]
            elif self.n_features != data.shape[1]:
                raise ValueError(f"Inconsistent feature dimensions: {self.n_features} vs {data.shape[1]} in {file_path}")
        
        self.time_points = self.max_rows
        print(f"Data dimensions: {self.time_points} time points × {self.n_features} features")
        
        # Compute normalization parameters if needed
        if normalize:
            print("Computing normalization parameters...")
            self.scaler = RobustScaler()
            all_data = []
            
            # Sample a subset for efficiency if dataset is large
            sample_size = min(100, len(self.files))
            for file_path, _ in tqdm(self.files[:sample_size]):
                data_dict = loadmat(file_path)
                key = self.file_keys[file_path]
                data = data_dict[key]
                
                # Pad if necessary
                if data.shape[0] < self.max_rows:
                    padding = np.zeros((self.max_rows - data.shape[0], data.shape[1]))
                    data = np.vstack((data, padding))
                elif data.shape[0] > self.max_rows:
                    data = data[:self.max_rows, :]
                
                all_data.append(data)
            
            # Fit scaler on concatenated data
            self.scaler.fit(np.vstack(all_data))
            self.normalize = normalize
            
            # Save normalization parameters
            joblib.dump({'scaler': self.scaler}, 'models/normalization_params.pkl')
        else:
            self.normalize = False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        data_dict = loadmat(file_path)
        key = self.file_keys[file_path]
        data = data_dict[key]
        
        # Pad or truncate to consistent dimensions
        if data.shape[0] < self.max_rows:
            padding = np.zeros((self.max_rows - data.shape[0], data.shape[1]))
            data = np.vstack((data, padding))
        elif data.shape[0] > self.max_rows:
            data = data[:self.max_rows, :]
        
        # Apply normalization
        if self.normalize:
            data = self.scaler.transform(data)
        
        # Convert to tensor
        data = torch.FloatTensor(data)
        
        if self.transform:
            data = self.transform(data)
            
        return data.flatten(), torch.tensor(label, dtype=torch.float32)

    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        labels = [label for _, label in self.files]
        class_counts = np.bincount(labels)
        return 1.0 / class_counts

class EnsembleModel(nn.Module):
    """Balanced ensemble with adjustable weights"""
    def __init__(self, models, device):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Initialize with balanced weights that sum to 1.0
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        self.device = device
        
        # Freeze base models' parameters - we don't want to train them again
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        predictions = torch.zeros(x.size(0), 1).to(self.device)
        
        # Gather predictions from each model - no torch.no_grad() here
        for i, model in enumerate(self.models):
            predictions += self.weights[i] * model(x)
        
        # Apply sigmoid for balanced output range
        return torch.sigmoid(predictions)

def train_model(model, train_loader, val_loader, device, epochs=10, class_weights=None):
    """Training function with bias monitoring"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Use weighted BCE loss for class imbalance
    if class_weights is not None:
        weight = torch.tensor([class_weights[0], class_weights[1]], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[1]/class_weights[0]], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_predictions = []  # Store predictions for bias analysis
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                predicted = (outputs.view(-1) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for bias analysis
                all_preds.extend(outputs.view(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_predictions.append((all_preds, all_labels))
        
        # Print metrics and analyze bias
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Bias analysis
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        adhd_preds = all_preds[all_labels == 1]
        control_preds = all_preds[all_labels == 0]
        
        print("  Bias Analysis:")
        print(f"    ADHD samples: {len(adhd_preds)}, Avg prediction: {np.mean(adhd_preds):.4f}")
        print(f"    Control samples: {len(control_preds)}, Avg prediction: {np.mean(control_preds):.4f}")
        print(f"    Prediction gap: {np.mean(adhd_preds) - np.mean(control_preds):.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            print("  New best model saved!")
            
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_predictions': val_predictions,
        'best_state_dict': best_state_dict,
        'best_val_loss': best_val_loss
    }

def create_and_train_models(adhd_folder, control_folder, batch_size=32, epochs=20):
    """Create and train individual models with bias control"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset with improved preprocessing
    dataset = BalancedMATDataset(adhd_folder, control_folder)
    
    # Split dataset with stratification to maintain class balance
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=[label for _, label in dataset.files],
        random_state=42
    )
    
    # Calculate class weights for balanced training
    class_weights = dataset.get_class_weights()
    print(f"Class weights: Control={class_weights[0]:.4f}, ADHD={class_weights[1]:.4f}")
    
    # Create samplers for balanced batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=batch_size, 
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=val_sampler
    )
    
    # Model input dimension
    input_dim = dataset.time_points * dataset.n_features
    
    # Create models with similar embed_dim for compatibility
    embed_dim = 256
    model_results = {}
    
    # Train linear model
    print("\nTraining Linear Model...")
    linear_model = LinearRegressionNet(input_dim=input_dim, embed_dim=embed_dim).to(device)
    model_results['linear'] = train_model(
        linear_model, train_loader, val_loader, device, 
        epochs=epochs, class_weights=class_weights
    )
    torch.save({
        'state_dict': model_results['linear']['best_state_dict'],
        'architecture': {'input_dim': input_dim, 'embed_dim': embed_dim}
    }, 'models/linear_balanced.pth')
    
    # Train tree model
    print("\nTraining Tree Model...")
    tree_model = TreeBasedNet(input_dim=input_dim, embed_dim=embed_dim).to(device)
    model_results['tree'] = train_model(
        tree_model, train_loader, val_loader, device, 
        epochs=epochs, class_weights=class_weights
    )
    torch.save({
        'state_dict': model_results['tree']['best_state_dict'],
        'architecture': {'input_dim': input_dim, 'embed_dim': embed_dim}
    }, 'models/tree_balanced.pth')
    
    # Train GBM model
    print("\nTraining GBM Model...")
    gbm_model = GBMNet(input_dim=input_dim, embed_dim=embed_dim).to(device)
    model_results['gbm'] = train_model(
        gbm_model, train_loader, val_loader, device, 
        epochs=epochs, class_weights=class_weights
    )
    torch.save({
        'state_dict': model_results['gbm']['best_state_dict'],
        'architecture': {'input_dim': input_dim, 'embed_dim': embed_dim}
    }, 'models/gbm_balanced.pth')
    
    # Create and train ensemble
    print("\nTraining Linear+Tree Ensemble...")
    linear_model = LinearRegressionNet(input_dim=input_dim, embed_dim=embed_dim).to(device)
    linear_model.load_state_dict(model_results['linear']['best_state_dict'])
    linear_model.eval()  # Set to evaluation mode
    
    tree_model = TreeBasedNet(input_dim=input_dim, embed_dim=embed_dim).to(device)
    tree_model.load_state_dict(model_results['tree']['best_state_dict'])
    tree_model.eval()  # Set to evaluation mode
    
    ensemble_model = EnsembleModel([linear_model, tree_model], device).to(device)
    model_results['linear_tree_ensemble'] = train_model(
        ensemble_model, train_loader, val_loader, device, 
        epochs=10, class_weights=class_weights
    )
    torch.save({
        'state_dict': model_results['linear_tree_ensemble']['best_state_dict'],
        'architecture': {
            'input_dim': input_dim, 
            'embed_dim': embed_dim,
            'models': ['linear', 'tree']
        }
    }, 'models/linear_tree_ensemble_balanced.pth')
    
    # Create and train another ensemble
    print("\nTraining Linear+GBM Ensemble...")
    linear_model = LinearRegressionNet(input_dim=input_dim, embed_dim=embed_dim).to(device)
    linear_model.load_state_dict(model_results['linear']['best_state_dict'])
    linear_model.eval()  # Set to evaluation mode
    
    gbm_model = GBMNet(input_dim=input_dim, embed_dim=embed_dim).to(device)
    gbm_model.load_state_dict(model_results['gbm']['best_state_dict'])
    gbm_model.eval()  # Set to evaluation mode
    
    ensemble_model = EnsembleModel([linear_model, gbm_model], device).to(device)
    model_results['linear_gbm_ensemble'] = train_model(
        ensemble_model, train_loader, val_loader, device, 
        epochs=10, class_weights=class_weights
    )
    torch.save({
        'state_dict': model_results['linear_gbm_ensemble']['best_state_dict'],
        'architecture': {
            'input_dim': input_dim, 
            'embed_dim': embed_dim,
            'models': ['linear', 'gbm']
        }
    }, 'models/linear_gbm_ensemble_balanced.pth')
    
    # Plot training results
    plot_training_results(model_results)
    
    # Analyze bias in predictions
    analyze_model_bias(model_results)
    
    return model_results

def plot_training_results(model_results):
    """Plot loss and accuracy curves for all models"""
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    for model_name, results in model_results.items():
        plt.plot(results['train_losses'], label=model_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation losses
    plt.subplot(2, 2, 2)
    for model_name, results in model_results.items():
        plt.plot(results['val_losses'], label=model_name)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracies
    plt.subplot(2, 2, 3)
    for model_name, results in model_results.items():
        plt.plot(results['val_accuracies'], label=model_name)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_results.png')
    plt.close()

def analyze_model_bias(model_results):
    """Analyze prediction bias in all models"""
    plt.figure(figsize=(15, 10))
    
    model_names = list(model_results.keys())
    adhd_means = []
    control_means = []
    
    for i, model_name in enumerate(model_names):
        # Get predictions from last epoch
        preds, labels = model_results[model_name]['val_predictions'][-1]
        preds = np.array(preds)
        labels = np.array(labels)
        
        # Separate predictions by class
        adhd_preds = preds[labels == 1]
        control_preds = preds[labels == 0]
        
        # Store means for bar chart
        adhd_means.append(np.mean(adhd_preds))
        control_means.append(np.mean(control_preds))
        
        # Plot histograms
        plt.subplot(len(model_names), 2, i*2+1)
        plt.hist(adhd_preds, bins=20, alpha=0.7, label='ADHD')
        plt.hist(control_preds, bins=20, alpha=0.7, label='Control')
        plt.title(f'{model_name} - Prediction Distribution')
        plt.xlabel('Prediction Value')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot ROC curve
        plt.subplot(len(model_names), 2, i*2+2)
        plot_decision_boundary(preds, labels, model_name)
    
    plt.tight_layout()
    plt.savefig('models/prediction_analysis.png')
    plt.close()
    
    # Create bar chart comparing ADHD vs Control means
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, adhd_means, width, label='ADHD')
    plt.bar(x + width/2, control_means, width, label='Control')
    
    plt.xlabel('Model')
    plt.ylabel('Mean Prediction')
    plt.title('Mean Prediction by Class')
    plt.xticks(x, model_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/prediction_means.png')
    plt.close()

def plot_decision_boundary(preds, labels, model_name):
    """Plot ROC and precision-recall curves"""
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()

if __name__ == "__main__":
    # Make models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Path to your data folders
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    
    # Train all models
    results = create_and_train_models(adhd_folder, control_folder, batch_size=32, epochs=20)
    
    print("\nTraining complete! Models saved to the 'models' directory.")
    print("Key results:")
    for model_name, results in results.items():
        print(f"  {model_name}: Best validation loss = {results['best_val_loss']:.4f}")
