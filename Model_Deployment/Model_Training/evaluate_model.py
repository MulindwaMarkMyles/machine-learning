import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import MATDataset, EEGNet
import joblib
import os
from torch.utils.data import DataLoader
import pandas as pd
from scipy.stats import spearmanr
import shap
import torch.nn as nn

def load_model_and_scaler(model_path, config_path, scaler_path, device='cpu'):
    """Load the trained model, configuration, and scaler"""
    try:
        # Load the config first
        config = joblib.load(config_path)
        
        # Load the scaler
        scaler = joblib.load(scaler_path)
        print("Loaded scaler successfully")
        
        # Load the state dict to check its architecture
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine the embedding dimension from the state dict
        features_weight = state_dict['features.0.weight']
        embed_dim = features_weight.size(0)  # First dimension of the first layer's weight
        print(f"Detected embedding dimension from saved model: {embed_dim}")
        
        # Update config with the correct embedding dimension
        config['embed_dim'] = embed_dim
        print(f"Updated config with embed_dim: {embed_dim}")
        
        # Create model with matching architecture
        model = EEGNet(
            input_dim=config['input_dim'],
            embed_dim=embed_dim,  # Use the detected dimension
            architecture_type='light' if embed_dim == 64 else 'standard' if embed_dim == 128 else 'deep' if embed_dim == 256 else 'mini'
        )
        
        # Load the state dict
        try:
            model.load_state_dict(state_dict)
            print("Model loaded successfully with exact architecture match")
        except Exception as e:
            print(f"Warning: Could not load state dict directly: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            
        model.to(device)
        model.eval()
        return model, config, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        raise

def evaluate_model(model, test_loader, device, config):
    """Evaluate model and return predictions and true labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            try:
                # Reshape data to match model's expected input
                batch_size = data.size(0)
                data = data.view(batch_size, -1)  # Flatten the data
                expected_dim = config['input_dim']
                current_dim = data.shape[1]
                
                # Handle dimension mismatch
                if current_dim != expected_dim:
                    if current_dim < expected_dim:
                        # Pad with zeros
                        padding = torch.zeros(batch_size, expected_dim - current_dim)
                        data = torch.cat([data, padding], dim=1)
                    else:
                        # Truncate
                        data = data[:, :expected_dim]
                
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = model(data)
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds if isinstance(preds, np.ndarray) else [preds])
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs if isinstance(probs, np.ndarray) else [probs])
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
            
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and optionally save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot and optionally save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_feature_importance(model, test_loader, device, config):
    """Analyze feature importance using gradient-based methods"""
    model.eval()
    num_features = 19  # Number of EEG channels
    feature_importance = np.zeros(num_features)
    num_samples = 0
    
    for data, _ in test_loader:
        try:
            batch_size = data.size(0)
            num_samples += batch_size
            
            # Reshape data for the model
            flattened_data = data.reshape(batch_size, -1)
            
            # Handle dimension mismatch
            if flattened_data.shape[1] != config['input_dim']:
                if flattened_data.shape[1] < config['input_dim']:
                    padding = torch.zeros(batch_size, config['input_dim'] - flattened_data.shape[1])
                    flattened_data = torch.cat([flattened_data, padding], dim=1)
                else:
                    flattened_data = flattened_data[:, :config['input_dim']]
            
            flattened_data = flattened_data.to(device)
            flattened_data.requires_grad = True
            
            outputs = model(flattened_data)
            outputs.sum().backward()
            
            # Calculate importance per channel by averaging across time
            gradients = flattened_data.grad.view(batch_size, -1, num_features)
            channel_importance = gradients.abs().mean(dim=(0, 1)).cpu().numpy()
            feature_importance += channel_importance
            
        except Exception as e:
            print(f"Error in feature importance calculation: {e}")
            continue
    
    return feature_importance / num_samples if num_samples > 0 else feature_importance

def plot_feature_importance(feature_importance, save_path=None):
    """Plot feature importance"""
    plt.figure(figsize=(12, 6))
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
        'Pz', 'P4', 'T6', 'O1', 'O2'
    ]
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(len(sorted_idx)) + .5
    
    plt.barh(pos, feature_importance[sorted_idx])
    plt.yticks(pos, np.array(channel_names)[sorted_idx])
    plt.xlabel('Importance Score')
    plt.title('EEG Channel Importance Analysis')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_detailed_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues')
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add raw counts as text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.2, i+0.7, f'({cm[i,j]})', 
                    fontsize=9, color='black')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_feature_correlations(test_loader):
    """Analyze correlations between features"""
    all_data = []
    for data, _ in test_loader:
        # Reshape to 2D: (batch_size * time_steps, features)
        reshaped_data = data.view(-1, data.shape[-1]).numpy()
        all_data.append(reshaped_data)
    
    # Concatenate all batches
    all_data = np.concatenate(all_data, axis=0)
    
    # Calculate correlation matrix between features
    corr_matrix = np.corrcoef(all_data.T)
    return corr_matrix

def plot_correlation_matrix(corr_matrix, save_path=None):
    """Plot feature correlation matrix"""
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='coolwarm', 
                center=0, 
                annot=True, 
                fmt='.2f',
                square=True,
                vmin=-1, 
                vmax=1)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

class ModelWrapper(nn.Module):
    """Wrapper class to make model more interpretable for SHAP"""
    def __init__(self, model, num_features=19, input_dim=None):
        super().__init__()
        self.model = model
        self.num_features = num_features
        self.input_dim = input_dim
    
    def forward(self, x):
        batch_size = x.size(0)
        # Reshape to match the expected input dimension
        x = x.view(batch_size, -1)  # Flatten completely
        if x.shape[1] != self.input_dim:
            if x.shape[1] < self.input_dim:
                padding = torch.zeros(batch_size, self.input_dim - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.input_dim]
        return self.model(x)

def generate_shap_analysis(model, test_loader, device, config):
    """Generate SHAP values for model interpretability with improved handling"""
    try:
        model.eval()  # Ensure model is in eval mode
        
        # Prepare test samples (use very few samples)
        test_batch = next(iter(test_loader))[0][:5]  # Analyze only 5 samples
        test_batch = test_batch[:, :1000, :] if test_batch.shape[1] > 1000 else test_batch
        
        # Initialize channel importance array
        channel_importance = np.zeros(19)
        num_samples = test_batch.size(0)
        expected_dim = config['input_dim']  # Get required input dimension
        
        for i in range(num_samples):
            # Create input tensor that requires gradient
            input_data = test_batch[i:i+1].clone().detach().to(device)
            input_data.requires_grad_(True)
            
            # Reshape and adjust dimensions
            flattened_input = input_data.reshape(1, -1)
            current_dim = flattened_input.shape[1]
            
            # Handle dimension mismatch
            if current_dim != expected_dim:
                if current_dim < expected_dim:
                    padding = torch.zeros(1, expected_dim - current_dim, device=device)
                    flattened_input = torch.cat([flattened_input, padding], dim=1)
                else:
                    flattened_input = flattened_input[:, :expected_dim]
            
            # Forward pass
            output = model(flattened_input)
            
            # Compute gradients for each output dimension
            output.backward(torch.ones_like(output))
            
            if input_data.grad is not None:
                # Reshape gradients back to original shape
                grads = input_data.grad.view(1, -1, 19)  # [1, time_steps, channels]
                
                # Compute importance per channel
                # Take absolute gradients and average across time steps
                importance = grads.abs().mean(dim=1).squeeze().cpu().numpy()
                channel_importance += importance
            
            # Clear gradients for next iteration
            if input_data.grad is not None:
                input_data.grad.zero_()
        
        # Average over samples
        channel_importance /= num_samples
        
        # Normalize importance scores
        if channel_importance.max() > 0:  # Avoid division by zero
            channel_importance = (channel_importance - channel_importance.min()) / \
                               (channel_importance.max() - channel_importance.min())
        
        return channel_importance, None  # Feature data not used in current implementation
        
    except Exception as e:
        print(f"Error in channel importance analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
    
def plot_shap_summary(shap_values, feature_data, save_path=None):
    """Plot SHAP summary with improved visualization"""
    if shap_values is None:
        return
    
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
        'Pz', 'P4', 'T6', 'O1', 'O2'
    ]
    
    # Sort channels by importance
    sorted_indices = np.argsort(shap_values)
    sorted_importance = shap_values[sorted_indices]
    sorted_names = np.array(channel_names)[sorted_indices]
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(channel_names))
    bars = plt.barh(y_pos, sorted_importance)
    plt.yticks(y_pos, sorted_names)
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.xlabel('Normalized Channel Importance')
    plt.title('EEG Channel Importance Analysis')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model and configuration
        model_path = "./models/best_model_ten.pth"
        config_path = "./models/model_config_ten.pkl"
        scaler_path = "./models/model_scaler_ten.pkl"
        
        # First, examine the saved model's architecture
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        features_weight = state_dict['features.0.weight']
        embed_dim = features_weight.size(0)
        print(f"\nSaved model architecture:")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Input dimension: {features_weight.size(1)}")
        
        model, config, scaler = load_model_and_scaler(model_path, config_path, scaler_path, device)
        
        # Print model structure for debugging
        print("\nLoaded model structure:")
        print(model)
        
        # Create test dataset with the loaded scaler
        adhd_folder = "../ADHD_part1/ADHD_part1"
        control_folder = "../Control_part1/Control_part1"
        test_dataset = MATDataset(adhd_folder, control_folder)
        
        # Replace the dataset's scaler with the loaded one
        test_dataset.scaler = scaler
        test_dataset.fitted_scaler = True
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Print dataset information
        print("\nDataset Information:")
        sample_data = test_dataset[0][0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Expected flattened dimension: {np.prod(sample_data.shape)}")
        
        # Print model and data dimensions for debugging
        print("\nDimension Information:")
        sample_data = next(iter(test_loader))[0]
        print(f"Model input dimension: {config['input_dim']}")
        print(f"Sample batch shape: {sample_data.shape}")
        print(f"Sample flattened dimension: {np.prod(sample_data.shape[1:])}")
        
        # Evaluate model
        predictions, true_labels, probabilities = evaluate_model(model, test_loader, device, config)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        # Create results directory if it doesn't exist
        os.makedirs("./results", exist_ok=True)
        
        # Plot and save confusion matrix
        plot_detailed_confusion_matrix(true_labels, predictions, "./results/detailed_confusion_matrix.png")
        
        # Plot and save ROC curve
        plot_roc_curve(true_labels, probabilities, "./results/roc_curve.png")
        
        # Plot and save precision-recall curve
        plot_precision_recall_curve(true_labels, probabilities, "./results/precision_recall_curve.png")
        
        # Feature importance analysis with proper error handling
        try:
            print("\nCalculating feature importance...")
            feature_importance = analyze_feature_importance(model, test_loader, device, config)
            plot_feature_importance(feature_importance, "./results/feature_importance.png")
        except Exception as e:
            print(f"Skipping feature importance analysis: {e}")
        
        # Correlation analysis with proper error handling
        try:
            corr_matrix = analyze_feature_correlations(test_loader)
            plot_correlation_matrix(corr_matrix, "./results/feature_correlations.png")
        except Exception as e:
            print(f"Skipping correlation analysis: {e}")
        
        # SHAP analysis with improved handling
        try:
            print("\nGenerating channel importance analysis...")
            channel_importance, feature_data = generate_shap_analysis(model, test_loader, device, config)
            if channel_importance is not None:
                # Plot and save channel importance
                plot_shap_summary(channel_importance, feature_data, "./results/channel_importance.png")
                
                # Save processed values
                np.save("./results/channel_importance.npy", channel_importance)
                
                # Print channel importance summary
                channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                               'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
                               'Pz', 'P4', 'T6', 'O1', 'O2']
                
                print("\nChannel Importance Ranking:")
                importance_dict = dict(zip(channel_names, channel_importance))
                for channel, importance in sorted(importance_dict.items(), 
                                               key=lambda x: x[1], reverse=True):
                    print(f"{channel}: {importance:.4f}")
                
                # Add average importance by brain region
                regions = {
                    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
                    'Central': ['C3', 'Cz', 'C4'],
                    'Temporal': ['T3', 'T4', 'T5', 'T6'],
                    'Parietal': ['P3', 'Pz', 'P4'],
                    'Occipital': ['O1', 'O2']
                }
                
                print("\nBrain Region Importance:")
                for region, channels in regions.items():
                    region_importance = np.mean([importance_dict[ch] for ch in channels])
                    print(f"{region}: {region_importance:.4f}")
    
        except Exception as e:
            print(f"Channel importance analysis failed: {e}")
        
        # Save detailed results
        results = {
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'feature_importance': feature_importance,
            'correlation_matrix': corr_matrix
        }
        joblib.dump(results, "./results/evaluation_results.pkl")
        
        # Calculate and print additional metrics
        accuracy = (predictions == true_labels).mean()
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Generate HTML report
        generate_html_report(results, config)
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_html_report(results, config):
    """Generate an HTML report with all results"""
    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ margin: 10px 0; }}
            img {{ max-width: 100%; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <div class="metric">
            <h2>Model Configuration</h2>
            <pre>
Input dimension: {config['input_dim']}
Embedding dimension: {config['embed_dim']}
Architecture type: {config.get('architecture_type', 'standard')}
            </pre>
        </div>
        <div class="metric">
            <h2>Performance Metrics</h2>
            <img src="detailed_confusion_matrix.png" alt="Confusion Matrix">
            <img src="roc_curve.png" alt="ROC Curve">
            <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
        </div>
        <div class="metric">
            <h2>Feature Analysis</h2>
            <img src="feature_importance.png" alt="Feature Importance">
            <img src="feature_correlations.png" alt="Feature Correlations">
            <img src="shap_summary.png" alt="SHAP Summary">
        </div>
    </body>
    </html>
    """
    
    with open("./results/evaluation_report.html", "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()
