import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
import shap
import matplotlib.pyplot as plt
import torch.nn as nn

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
    
    def predict(self, x):
        """Predict method for sklearn/shap compatibility"""
        if isinstance(x, np.ndarray):
            # Get the expected input dimension from first linear layer
            first_linear = None
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            input_dim = first_linear.in_features if first_linear else None
            
            # Pad or reshape input if needed
            if x.shape[1] != input_dim:
                x = np.pad(x, ((0, 0), (0, input_dim - x.shape[1])), mode='constant')
            x = torch.FloatTensor(x)
            
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy()

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

def analyze_shap():
    print("Starting SHAP analysis...")
    
    print("Loading dataset...")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    print("Preparing data...")
    X, y = prepare_data(dataset)
    
    # Find most important features first
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select a reasonable number of features
    n_features = 20
    print(f"Selecting top {n_features} most informative features...")
    selector = SelectKBest(f_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    
    # Get indices of selected features
    selected_indices = selector.get_support(indices=True)
    feature_names = [f"feature_{i}" for i in selected_indices]
    
    # Use only selected features
    X = X[:, selected_indices]
    
    # Use a reasonable sample size
    n_samples = 100  # Analyze 100 samples for better representation
    X = X[:n_samples]
    y = y[:n_samples]
    
    # Free memory
    dataset = None
    
    print(f"Using {n_samples} samples and {n_features} features for analysis...")
    print(f"Data shape: {X.shape}")
    
    model_types = ['best', 'worst']
    for model_type in model_types:
        print(f"\nAnalyzing {model_type} models...")
        for i in range(1, 3):
            print(f"Processing {model_type} model {i}...")
            
            print("Loading model...")
            model, model_names, _ = load_ensemble_model(
                f'../Model_Training/models/{model_type}_ensemble_combination_{i}.pth',
            )
            
            print("Calculating SHAP values...")
            
            try:
                # Use more background samples for better representation
                background_samples = min(30, n_samples//2)
                background = shap.sample(X, background_samples)
                print(f"Background shape: {background.shape}")
                
                # Create explainer with fewer nsamples for efficiency
                explainer = shap.KernelExplainer(model.predict, background)
                
                # Analyze more samples for better insights
                analysis_samples = min(15, n_samples)  # Analyze more samples
                print(f"Analyzing {analysis_samples} samples...")
                
                shap_values = explainer.shap_values(
                    X[:analysis_samples],
                    nsamples=100  # More samples for better estimation
                )
                
                print("Generating and saving plots...")
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                if len(shap_values.shape) == 3:
                    shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])
                
                print(f"SHAP values shape: {shap_values.shape}")
                
                # Generate better summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values, 
                    X[:analysis_samples],
                    feature_names=feature_names,
                    show=False,
                    max_display=20  # Show all selected features
                )
                
                output_plot = f'shap_summary_{model_type}_{i}.png'
                plt.savefig(output_plot, bbox_inches='tight', dpi=300)
                print(f"Saved plot to {output_plot}")
                plt.close()
                
                # Create bar plot of mean absolute SHAP values
                plt.figure(figsize=(10, 6))
                shap_importance = np.abs(shap_values).mean(0)
                sorted_idx = shap_importance.argsort()
                plt.barh(
                    range(len(sorted_idx)), 
                    shap_importance[sorted_idx],
                    tick_label=[feature_names[i] for i in sorted_idx]
                )
                plt.title(f'Mean |SHAP| ({model_type.title()} Model {i})')
                plt.tight_layout()
                plt.savefig(f'shap_importance_{model_type}_{i}.png', dpi=300)
                plt.close()
                
                # Save more detailed SHAP values
                output_values = f'shap_values_{model_type}_{i}.npz'
                np.savez(
                    output_values,
                    shap_values=shap_values,
                    feature_names=feature_names,
                    feature_indices=selected_indices,
                    analysis_samples=X[:analysis_samples]
                )
                print(f"Saved SHAP values to {output_values}")
                
                # Clear model from memory
                model = None
                explainer = None
                
            except Exception as e:
                print(f"Error during SHAP analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    analyze_shap()
