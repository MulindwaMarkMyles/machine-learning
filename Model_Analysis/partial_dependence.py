import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import torch.nn as nn
from model_classes import *

class EnsembleModel(nn.Module):
    def __init__(self, models, device):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        self.device = device

    def forward(self, x):
        predictions = torch.zeros(x.size(0), 1).to(self.device)
        for i, model in enumerate(self.models):
            predictions += self.weights[i] * model(x)
        return torch.sigmoid(predictions)

class SklearnModelWrapper:
    """Wrapper to make PyTorch models compatible with sklearn's interface"""
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Get the expected input dimension from the first linear layer
        first_linear = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        self.input_dim = first_linear.in_features if first_linear else None
        self._fitted = True  # Pretend the model is already fitted
    
    def fit(self, X, y):
        # sklearn compatibility - pretend to fit the model
        return self
    
    def predict(self, X):
        # Ensure X is properly shaped for the model
        if len(X.shape) == 2 and X.shape[1] != self.input_dim:
            X = np.pad(X, ((0, 0), (0, self.input_dim - X.shape[1])), mode='constant')
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs.squeeze() > 0.5).int()
            
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        # Ensure X is properly shaped for the model
        if len(X.shape) == 2 and X.shape[1] != self.input_dim:
            X = np.pad(X, ((0, 0), (0, self.input_dim - X.shape[1])), mode='constant')
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
            
        # Return probabilities for both classes [P(class=0), P(class=1)]
        return np.hstack([1-outputs, outputs])

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

def prepare_data(dataset):
    X = []
    y = []
    for data, label in dataset:
        X.append(data.numpy().flatten())
        y.append(label)
    return np.array(X), np.array(y)

def get_feature_names(dataset):
    return [f"feature_{i}" for i in range(dataset[0][0].numel())]

def analyze_pdp():
    print("Starting partial dependence analysis...")
    
    print("Loading dataset...")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    print("Preparing data...")
    X, y = prepare_data(dataset)
    print(f"Original data shape: {X.shape}")
    
    # Use feature selection to identify most informative features
    n_features = 20  # Select top 20 features
    print(f"Selecting top {n_features} most informative features...")
    selector = SelectKBest(f_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    
    # Get indices of selected features
    selected_indices = selector.get_support(indices=True)
    feature_names = [f"feature_{i}" for i in selected_indices]
    
    # Extract only those features
    X = X[:, selected_indices]
    
    # Use a more reasonable sample size
    n_samples = 30
    X = X[:n_samples]
    y = y[:n_samples]
    
    # Free up memory
    dataset = None
    
    print(f"Reduced data shape: {X.shape}")
    print(f"Using {n_samples} samples and {n_features} features for analysis...")
    
    # Compute feature importance scores to identify top features for PDP
    from sklearn.ensemble import RandomForestClassifier
    print("Computing feature importances...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    # Select top 6 features for PDP based on importance
    top_features_idx = np.argsort(importances)[-6:]
    top_feature_names = [feature_names[i] for i in top_features_idx]
    print(f"Selected features for PDP: {top_feature_names}")
    
    model_types = ['best', 'worst']
    for model_type in model_types:
        print(f"\nAnalyzing {model_type} models...")
        for i in range(1, 3):
            try:
                print(f"Processing {model_type} model {i}...")
                
                print("Loading model...")
                model, model_names, _ = load_ensemble_model(
                    f'../Model_Training/models/{model_type}_ensemble_combination_{i}.pth'
                )
                
                # Wrap model for sklearn compatibility
                wrapped_model = SklearnModelWrapper(model)
                
                # Use a different approach to calculate partial dependence manually
                print("Calculating partial dependence plots...")
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()
                
                # Define a manual pdp function since sklearn's partial_dependence has issues
                def manual_pdp(model, X, feature_idx, grid_points=20):
                    feature_values = X[:, feature_idx]
                    min_val = np.min(feature_values)
                    max_val = np.max(feature_values)
                    
                    # Create grid values
                    grid = np.linspace(min_val, max_val, grid_points)
                    pdp_values = []
                    
                    # Calculate predictions for each grid value
                    for value in grid:
                        X_modified = X.copy()
                        X_modified[:, feature_idx] = value
                        predictions = model.predict_proba(X_modified)[:, 1]  # Get probability of class 1
                        pdp_values.append(np.mean(predictions))
                    
                    return grid, np.array(pdp_values)
                
                for idx, feature_idx in enumerate(top_features_idx):
                    print(f"Processing feature {feature_names[feature_idx]}...")
                    try:
                        grid, pdp_values = manual_pdp(wrapped_model, X, feature_idx)
                        
                        # Plot PDP with improved formatting
                        axes[idx].plot(grid, pdp_values, 'b-', linewidth=2)
                        axes[idx].set_title(f'PDP for {feature_names[feature_idx]}')
                        axes[idx].set_xlabel(feature_names[feature_idx])
                        axes[idx].set_ylabel('Predicted probability (ADHD)')
                        axes[idx].grid(True, linestyle='--', alpha=0.5)
                    except Exception as e:
                        print(f"Error processing feature {feature_names[feature_idx]}: {str(e)}")
                
                plt.suptitle(f'Partial Dependence Plots ({model_type.title()} Model {i})')
                plt.tight_layout()
                
                output_file = f'pdp_analysis_{model_type}_{i}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {output_file}")
                plt.close()
                
                # Save PDP data for future reference
                output_data = f'pdp_data_{model_type}_{i}.npz'
                pdp_values_dict = {}
                pdp_grid_dict = {}
                
                try:
                    for feature_idx in top_features_idx:
                        grid, values = manual_pdp(wrapped_model, X, feature_idx)
                        pdp_values_dict[feature_names[feature_idx]] = values
                        pdp_grid_dict[feature_names[feature_idx]] = grid
                    
                    np.savez(
                        output_data,
                        feature_names=np.array(top_feature_names),
                        feature_indices=top_features_idx,
                        pdp_values=pdp_values_dict,
                        pdp_grid=pdp_grid_dict
                    )
                    print(f"Saved PDP data to {output_data}")
                except Exception as e:
                    print(f"Error saving PDP data: {str(e)}")
                
                # Clean up memory
                model = None
                wrapped_model = None
                
            except Exception as e:
                print(f"Error during analysis of {model_type} model {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    analyze_pdp()
