import torch
import torch.nn as nn
import json
from multimodel_train import (MATDataset, LinearRegressionNet, LogisticRegressionNet,
                            TreeBasedNet, SVMNet, KNNNet, GBMNet, XGBoostNet, 
                            DeepNeuralNet, PCANet)
from torch.utils.data import DataLoader
import joblib
import numpy as np
from itertools import combinations
from typing import List, Tuple
from tqdm import tqdm

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

def load_model(model_name, input_dim, config_path, weights_path):
    """Load model with proper configuration and weights"""
    try:
        # Load weights with proper handling of nested state dict
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Get the state dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'architecture' in checkpoint:
                arch = checkpoint['architecture']
                input_dim = arch['input_dim']
                embed_dim = arch['embed_dim']
                state_dict = checkpoint.get('state_dict', checkpoint)
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Extract embed_dim from the state dict if not already set
        if 'embed_dim' not in locals():
            # Get embed_dim from the first layer's weight shape
            first_layer_weight = next(key for key in state_dict.keys() if 'weight' in key)
            embed_dim = state_dict[first_layer_weight].shape[0]
            print(f"Extracted embed_dim={embed_dim} from weights")

        model_classes = {
            'linear': LinearRegressionNet,
            'logistic': LogisticRegressionNet,
            'tree': TreeBasedNet,
            'svm': SVMNet,
            'knn': KNNNet,
            'gbm': GBMNet,
            'xgboost': XGBoostNet,
            'neural': DeepNeuralNet,
            'pca': PCANet
        }
        
        # Create model using extracted dimensions
        if model_name == 'pca':
            model = model_classes[model_name](
                input_dim=input_dim,
                embed_dim=embed_dim,
                n_components=embed_dim//2
            )
        else:
            model = model_classes[model_name](
                input_dim=input_dim,
                embed_dim=embed_dim
            )
        
        # Load the state dict
        model.load_state_dict(state_dict)
        print(f"Successfully loaded {model_name} model with embed_dim={embed_dim}")
        return model
        
    except Exception as e:
        print(f"Error loading {model_name} model: {str(e)}")
        print(f"Config path: {config_path}")
        print(f"Weights path: {weights_path}")
        raise

def evaluate_ensemble(ensemble: nn.Module, val_loader: DataLoader, device: torch.device) -> dict:
    """Evaluate ensemble performance with multiple metrics"""
    ensemble.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = ensemble(data)
            predicted = (outputs.squeeze() > 0.5).int()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays for metric calculations
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }

def train_ensemble_combination(models: List[nn.Module], 
                             model_names: List[str],
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             device: torch.device,
                             num_epochs: int = 10) -> Tuple[dict, torch.Tensor]:
    """Train an ensemble with a specific combination of models"""
    ensemble = EnsembleModel(models, device).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam([
        {'params': ensemble.models.parameters(), 'lr': 1e-4},
        {'params': ensemble.weights, 'lr': 1e-2}
    ])
    
    best_metrics = None
    best_weights = None
    
    for epoch in range(num_epochs):
        # Training
        ensemble.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = ensemble(data)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                ensemble.weights.data = torch.softmax(ensemble.weights.data, dim=0)
        
        # Validation
        current_metrics = evaluate_ensemble(ensemble, val_loader, device)
        
        # Update best metrics based on F1 score
        if best_metrics is None or current_metrics['f1'] > best_metrics['f1']:
            best_metrics = current_metrics
            best_weights = ensemble.weights.data.clone()
    
    return best_metrics, best_weights

def try_model_combinations(all_models: List[nn.Module],
                         model_names: List[str],
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         device: torch.device) -> List[dict]:
    """Try all possible pairs of models and track performance"""
    results = []
    
    # Get all possible pairs of models
    model_combinations = list(combinations(range(len(all_models)), 2))
    
    print(f"\nTrying all possible pairs of models ({len(model_combinations)} combinations)...")
    
    for combo_indices in tqdm(model_combinations):
        # Get the current pair of models
        current_models = [all_models[i] for i in combo_indices]
        current_names = [model_names[i] for i in combo_indices]
        
        # Train and evaluate this combination
        metrics, weights = train_ensemble_combination(
            current_models, current_names,
            train_loader, val_loader, device
        )
        
        # Store results
        results.append({
            'models': current_names,
            'metrics': metrics,
            'weights': weights.cpu().numpy()
        })
    
    # Sort results by different metrics
    sorted_results = {
        'by_accuracy': sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True),
        'by_precision': sorted(results, key=lambda x: x['metrics']['precision'], reverse=True),
        'by_recall': sorted(results, key=lambda x: x['metrics']['recall'], reverse=True),
        'by_f1': sorted(results, key=lambda x: x['metrics']['f1'], reverse=True),
        'by_specificity': sorted(results, key=lambda x: x['metrics']['specificity'], reverse=True)
    }
    
    return sorted_results

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load all models with error handling
    input_dim = dataset[0][0].shape[1] * dataset[0][0].shape[0]
    models = []
    model_names = ['linear', 'logistic', 'tree', 'svm', 'knn', 'gbm', 'xgboost', 'neural', 'pca']
    loaded_model_names = []  # Keep track of successfully loaded models
    
    for model_name in model_names:
        try:
            config_path = f'./models/{model_name}_config.pkl'
            weights_path = f'./models/{model_name}_best.pth'
            
            model = load_model(model_name, input_dim, config_path, weights_path)
            model.to(device)
            model.eval()
            models.append(model)
            loaded_model_names.append(model_name)
            
        except Exception as e:
            print(f"Skipping {model_name} due to error: {str(e)}")
            continue
    
    if not models:
        raise RuntimeError("No models were successfully loaded!")
    
    print(f"\nSuccessfully loaded {len(models)} models: {', '.join(loaded_model_names)}")
    
    # Use loaded_model_names instead of model_names for combinations
    results = try_model_combinations(models, loaded_model_names, train_loader, val_loader, device)
    
    # Save and display results for each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    print("\nTop 5 Model Combinations by Metric:")
    print("-" * 50)
    for metric in metrics:
        print(f"\nTop 5 by {metric.capitalize()}:")
        for i, result in enumerate(results[f'by_{metric}'][:5]):
            print(f"\n{i+1}. {metric.capitalize()}: {result['metrics'][metric]:.4f}")
            print("Models:", ', '.join(result['models']))
            print("Weights:", ', '.join([f"{w:.3f}" for w in result['weights']]))
    
    # Save best and worst combinations
    for result_type in ['best', 'worst']:
        for i in range(2):  # Save top/bottom 2 models
            current_result = results['by_f1'][i] if result_type == 'best' else results['by_f1'][-i-1]
            current_indices = [loaded_model_names.index(name) for name in current_result['models']]
            current_models = [models[i] for i in current_indices]
            
            # Create and save ensemble
            current_ensemble = EnsembleModel(current_models, device).to(device)
            current_ensemble.weights.data = torch.tensor(current_result['weights'], device=device)
            
            # Save complete ensemble
            torch.save({
                'ensemble': current_ensemble,
                'model_names': current_result['models'],
                'metrics': current_result['metrics'],
                'weights': current_result['weights']
            }, f'./models/{result_type}_ensemble_combination_{i+1}.pth')
    
    # Save all results
    with open('./models/ensemble_combinations_results.json', 'w') as f:
        json.dump({
            'combinations': [{
                'models': r['models'],
                'metrics': r['metrics'],
                'weights': r['weights'].tolist()
            } for r in results['by_f1']]
        }, f, indent=4)
    
if __name__ == "__main__":
    main()
