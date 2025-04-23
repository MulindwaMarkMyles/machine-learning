import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from lime import lime_tabular
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
    
    def predict_proba(self, x):
        """Method for LIME compatibility"""
        if isinstance(x, np.ndarray):
            # Get input dimension from first linear layer
            first_linear = None
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            input_dim = first_linear.in_features if first_linear else None
            
            # Pad if needed
            if x.shape[1] != input_dim:
                x = np.pad(x, ((0, 0), (0, input_dim - x.shape[1])), mode='constant')
            x = torch.FloatTensor(x)
            
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x).cpu().numpy()
            # LIME expects [probability of negative class, probability of positive class]
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

def analyze_lime():
    print("Starting LIME analysis...")
    
    print("Loading dataset...")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    print("Preparing data...")
    X, y = prepare_data(dataset)
    print(f"Original data shape: {X.shape}")
    
    # Use feature selection to find most informative features
    n_features = 20  # Select top 20 features
    print(f"Selecting top {n_features} most informative features...")
    selector = SelectKBest(f_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    
    # Get indices and names of selected features
    selected_indices = selector.get_support(indices=True)
    feature_names = [f"feature_{i}" for i in selected_indices]
    
    # Extract only those features
    X = X[:, selected_indices]
    
    # Use a reasonable sample size
    n_samples = 30
    X = X[:n_samples]
    y = y[:n_samples]
    
    # Free up memory
    dataset = None
    
    print(f"Reduced data shape: {X.shape}")
    print(f"Using {n_samples} samples and {n_features} features for analysis...")
    
    print("Initializing LIME explainer...")
    explainer = lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=['Control', 'ADHD'],
        verbose=True,
        mode='classification'
    )
    
    model_types = ['best', 'worst']
    for model_type in model_types:
        print(f"\nAnalyzing {model_type} models...")
        
        # Create directory for this model type
        output_dir = f'lime_explanations_{model_type}'
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(1, 3):
            try:
                print(f"Processing {model_type} model {i}...")
                
                print("Loading model...")
                model, model_names, _ = load_ensemble_model(
                    f'../Model_Training/models/{model_type}_ensemble_combination_{i}.pth'
                )
                
                # Generate explanations for select samples
                num_samples_to_explain = 5
                print(f"Explaining {num_samples_to_explain} samples...")
                
                # Create a summary plot showing feature importance across samples
                # Using a dictionary instead of direct indexing to handle transformed feature names
                feature_importance_dict = {name: 0.0 for name in feature_names}
                
                for j in range(num_samples_to_explain):
                    # Pick a sample
                    print(f"  Analyzing sample {j+1}/{num_samples_to_explain}...")
                    sample_idx = j
                    
                    # Generate explanation
                    exp = explainer.explain_instance(
                        X[sample_idx], 
                        model.predict_proba,
                        num_features=min(10, n_features),
                        num_samples=100
                    )
                    
                    # Save HTML explanation
                    output_html = f'{output_dir}/explanation_model{i}_sample{j}.html'
                    exp.save_to_file(output_html)
                    print(f"  Saved HTML explanation to {output_html}")
                    
                    # Save as image too
                    plt.figure(figsize=(10, 6))
                    exp.as_pyplot_figure(label=1)  # Focus on ADHD class
                    output_img = f'{output_dir}/explanation_model{i}_sample{j}.png'
                    plt.savefig(output_img, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # Save explanation data for later analysis
                    explanation_data = {
                        'feature_values': X[sample_idx],
                        'prediction': model.predict_proba(X[sample_idx].reshape(1, -1))[0, 1],
                        'true_label': y[sample_idx],
                        'explanation': exp.as_list(label=1)
                    }
                    output_exp_data = f'{output_dir}/explanation_data_model{i}_sample{j}.npz'
                    np.savez(output_exp_data, **explanation_data)
                    
                    # Track feature importance - with proper handling of transformed features
                    for feature_name_with_condition, weight in exp.as_list(label=1):
                        # Extract base feature name by removing any conditions (e.g., "feature_123 <= 0.5" -> "feature_123")
                        base_feature_name = feature_name_with_condition.split()[0]
                        
                        # Find the corresponding feature in our selected features
                        matching_features = [f for f in feature_names if f == base_feature_name]
                        if matching_features:
                            # Update importance for the matching feature
                            feature_importance_dict[matching_features[0]] += abs(weight)
                        else:
                            print(f"  Warning: Feature {base_feature_name} not found in selected features")
                
                # Convert dictionary to arrays for plotting
                all_feature_names = list(feature_importance_dict.keys())
                all_feature_weights = np.array([feature_importance_dict[f] for f in all_feature_names])
                
                # Normalize weights
                if num_samples_to_explain > 0:
                    all_feature_weights /= num_samples_to_explain
                
                # Plot overall feature importance
                plt.figure(figsize=(10, 6))
                if len(all_feature_weights) > 0:
                    sorted_idx = np.argsort(all_feature_weights)
                    plt.barh(
                        range(len(all_feature_names)), 
                        all_feature_weights[sorted_idx],
                        tick_label=[all_feature_names[i] for i in sorted_idx]
                    )
                    plt.title(f'Average LIME Feature Importance\n({model_type.title()} Model {i})')
                    plt.tight_layout()
                    summary_path = f'{output_dir}/lime_summary_model{i}.png'
                    plt.savefig(summary_path, dpi=300)
                    plt.close()
                    print(f"Saved feature importance summary to {summary_path}")
                
                # Save numerical results
                output_data = f'{output_dir}/lime_importance_model{i}.npz'
                np.savez(
                    output_data,
                    feature_importance=all_feature_weights,
                    feature_names=np.array(all_feature_names),
                    feature_indices=selected_indices
                )
                print(f"Saved numerical results to {output_data}")
                
                # Clear model from memory
                model = None
                
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
    analyze_lime()
