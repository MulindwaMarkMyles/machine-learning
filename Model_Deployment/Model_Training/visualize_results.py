import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
from multimodel_train import MATDataset
from fine_tune_models import ModelFineTuner

def load_results():
    """Load the fine-tuning results from JSON"""
    with open('./models/fine_tuned_results.json', 'r') as f:
        return json.load(f)

def plot_performance_comparison(results):
    """Plot original vs fine-tuned performance for each model"""
    models = list(results.keys())
    original_scores = [results[m]['original_score'] for m in models]
    fine_tuned_scores = [results[m]['fine_tuned_score'] for m in models]
    improvements = [results[m]['improvement'] for m in models]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25

    plt.bar(x - width, original_scores, width, label='Original Score', color='skyblue')
    plt.bar(x, fine_tuned_scores, width, label='Fine-tuned Score', color='lightgreen')
    plt.bar(x + width, improvements, width, label='Improvement', color='salmon')

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./visualizations/performance_comparison.png')
    plt.close()

def plot_improvements_heatmap(results):
    """Create a heatmap of parameter improvements"""
    models = list(results.keys())
    params = list(results[models[0]]['best_params'].keys())
    
    # Create a matrix of parameter values
    param_matrix = np.zeros((len(models), len(params)))
    for i, model in enumerate(models):
        for j, param in enumerate(params):
            if param in results[model]['best_params']:
                param_matrix[i, j] = results[model]['best_params'][param]

    plt.figure(figsize=(12, 8))
    sns.heatmap(param_matrix, xticklabels=params, yticklabels=models, 
                annot=True, fmt='.2e', cmap='viridis')
    plt.title('Model Parameters Heatmap')
    plt.tight_layout()
    plt.savefig('./visualizations/parameter_heatmap.png')
    plt.close()

def load_model_weights(model, model_name):
    """Safely load model weights with proper error handling"""
    model_path = f'./models/{model_name}_optimized.pth'
    
    import os
    if not os.path.exists(model_path):
        print(f"Warning: Model weights file not found for {model_name} at {model_path}")
        return False
        
    try:
        checkpoint = torch.load(
            model_path,
            weights_only=False,  # Need to load architecture info
            map_location=torch.device('cpu')
        )
        
        # Verify architecture matches
        saved_arch = checkpoint['architecture']
        current_arch = {
            'input_dim': model.features[0].in_features,
            'embed_dim': model.features[0].out_features,
            'n_components': getattr(model, 'n_components', None)
        }
        
        if saved_arch != current_arch:
            print(f"Warning: Architecture mismatch for {model_name}")
            print(f"Saved: {saved_arch}")
            print(f"Current: {current_arch}")
            return False
            
        model.load_state_dict(checkpoint['state_dict'])
        return True
    except Exception as e:
        print(f"Error loading weights for {model_name}: {str(e)}")
        return False

def generate_confusion_matrices(dataset, results):
    """Generate confusion matrices for each model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for idx, (model_name, model_results) in enumerate(results.items()):
        # Create model with same architecture as saved model
        input_dim = dataset[0][0].shape[1] * dataset[0][0].shape[0]
        params = model_results['best_params']
        
        # Handle PCA special case
        if model_name == 'pca':
            max_components = min(input_dim, params['n_components'])
            if max_components < params['n_components']:
                print(f"Warning: PCA n_components={params['n_components']} is too large, reducing to {max_components}")
                params['n_components'] = max_components
        
        tuner = ModelFineTuner(dataset, device, model_name, params)
        model = tuner.create_model(input_dim)
        model.eval()
        
        # Load weights
        weights_loaded = load_model_weights(model, model_name)
        
        # Get predictions
        test_loader = torch.utils.data.DataLoader(tuner.test_data, 
                                                batch_size=32, 
                                                shuffle=False)
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                outputs = model(data)
                predictions = (outputs.squeeze() > 0.5).cpu().numpy()
                y_true.extend(labels.numpy())
                y_pred.extend(predictions)

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{model_name.upper()} Confusion Matrix' + 
                          (' (No weights)' if not weights_loaded else ''))
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('./visualizations/confusion_matrices.png')
    plt.close()

def plot_learning_curves(results):
    """Plot learning improvement trajectories"""
    plt.figure(figsize=(10, 6))
    
    for model_name, model_results in results.items():
        improvements = [0]  # Start from 0
        improvements.append(model_results['improvement'])
        
        plt.plot([0, 1], improvements, 'o-', label=model_name)

    plt.xlabel('Training Phase')
    plt.ylabel('Improvement Score')
    plt.title('Learning Improvement Trajectories')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./visualizations/learning_curves.png')
    plt.close()

def create_summary_report(results):
    """Create a summary report of the results"""
    with open('./visualizations/summary_report.txt', 'w') as f:
        f.write("Model Fine-tuning Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Best performing model
        best_model = max(results.items(), 
                        key=lambda x: x[1]['fine_tuned_score'])
        f.write(f"Best Performing Model: {best_model[0].upper()}\n")
        f.write(f"Score: {best_model[1]['fine_tuned_score']:.4f}\n\n")
        
        # Most improved model
        most_improved = max(results.items(), 
                          key=lambda x: x[1]['improvement'])
        f.write(f"Most Improved Model: {most_improved[0].upper()}\n")
        f.write(f"Improvement: {most_improved[1]['improvement']:.4f}\n\n")
        
        # Detailed results
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        for model_name, model_results in results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"Original Score: {model_results['original_score']:.4f}\n")
            f.write(f"Fine-tuned Score: {model_results['fine_tuned_score']:.4f}\n")
            f.write(f"Improvement: {model_results['improvement']:.4f}\n")

def main():
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs('./visualizations', exist_ok=True)
    os.makedirs('./models', exist_ok=True)  # Also ensure models directory exists
    
    # Load results
    results = load_results()
    
    # Load dataset for confusion matrices
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    # Generate all visualizations
    plot_performance_comparison(results)
    plot_improvements_heatmap(results)
    generate_confusion_matrices(dataset, results)
    plot_learning_curves(results)
    create_summary_report(results)

if __name__ == "__main__":
    main()
