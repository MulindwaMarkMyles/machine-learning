import torch
from torch.utils.data import Dataset
import numpy as np
import os
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import torch.nn as nn
from model_classes import * 

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


def load_ensemble_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
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

class SklearnModelWrapper:
    """Wrapper to make PyTorch models compatible with sklearn's interface"""
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Set to evaluation mode
        # Get the expected input dimension from the first linear layer
        first_linear = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        self.input_dim = first_linear.in_features if first_linear else None

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Ensure X is properly shaped for the model
        if len(X.shape) == 2 and X.shape[1] != self.input_dim:
            # Reshape X to match the model's expected input dimension
            X = np.pad(X, ((0, 0), (0, self.input_dim - X.shape[1])), mode='constant')
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs.squeeze() > 0.5).int()
        
        return predictions.cpu().numpy()
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

def analyze_permutation():
    print("Starting permutation importance analysis...")
    
    print("Loading dataset...")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    print("Preparing data...")
    X, y = prepare_data(dataset)
    print(f"Original data shape: {X.shape}")
    
    # Find most important features first
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select a reasonable number of features - not too many, not too few
    n_features = 20  # Increased from 5 to 20 for better analysis
    print(f"Selecting top {n_features} most informative features...")
    selector = SelectKBest(f_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    
    # Get indices of selected features
    selected_indices = selector.get_support(indices=True)
    feature_names = [f"feature_{i}" for i in selected_indices]
    
    # Extract only those features from original data
    X = X[:, selected_indices]
    
    # Use a reasonable number of samples for meaningful analysis
    n_samples = 30  # Increased from 10 to 30 for better analysis
    X = X[:n_samples]
    y = y[:n_samples]
    
    # Free up memory
    dataset = None
    
    print(f"Reduced data shape: {X.shape}")
    print(f"Using {n_samples} samples and {n_features} features for analysis...")
    
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
                
                print("Calculating permutation importance...")
                sklearn_model = SklearnModelWrapper(model)
                
                # More repeats for stability, but still memory efficient
                result = permutation_importance(
                    sklearn_model, X, y,
                    n_repeats=5,      # Increased from 2 to 5
                    n_jobs=1,         # Single job to conserve memory
                    random_state=42,
                    scoring='accuracy'
                )
                
                # Get the top 10 features by importance
                sorted_idx = result.importances_mean.argsort()
                top_k = min(10, n_features)
                important_indices = sorted_idx[-top_k:]
                
                print("\nTop 10 most important features:")
                for idx in important_indices[::-1]:  # Reverse order (highest first)
                    print(f"Feature {feature_names[idx]}: {result.importances_mean[idx]:.4f} "
                          f"± {result.importances_std[idx]:.4f}")
                
                print("\nGenerating plots...")
                # Create better visualization
                plt.figure(figsize=(10, 6))  # Larger figure for better readability
                
                # Bar chart instead of boxplot for clearer visualization
                importance = result.importances_mean[important_indices]
                std = result.importances_std[important_indices]
                features = [feature_names[i] for i in important_indices]
                
                # Sort by importance for better visualization
                sorted_order = importance.argsort()
                importance = importance[sorted_order]
                std = std[sorted_order]
                features = [features[i] for i in sorted_order]
                
                # Plot the sorted features
                y_pos = np.arange(len(features))
                plt.barh(y_pos, importance, xerr=std, align='center', alpha=0.8)
                plt.yticks(y_pos, features)
                plt.xlabel('Feature Importance')
                plt.title(f'Permutation Importance ({model_type.title()} Model {i})')
                plt.tight_layout()
                
                output_file = f'permutation_importance_{model_type}_{i}.png'
                plt.savefig(output_file, dpi=300)
                print(f"Saved plot to {output_file}")
                plt.close()
                
                # Clear model from memory
                model = None
                sklearn_model = None
                
                # Save detailed results
                output_data = f'permutation_importance_{model_type}_{i}.npz'
                np.savez(
                    output_data,
                    importances_mean=result.importances_mean,
                    importances_std=result.importances_std,
                    feature_names=feature_names,
                    feature_indices=selected_indices
                )
                print(f"Saved numerical results to {output_data}")
                
            except Exception as e:
                print(f"Error during analysis of {model_type} model {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Force garbage collection after each model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    analyze_permutation()
