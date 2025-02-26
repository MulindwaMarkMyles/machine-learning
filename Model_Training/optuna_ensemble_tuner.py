import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.trial import TrialState
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path so we can import the model classes
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(parent_dir))

# Import model classes from the Flask deployment directory
from Flask_Deploy.app import LinearRegressionNet, GBMNet, EnsembleModel

# Configuration
DATA_DIR = os.path.join(parent_dir, "data")
MODEL_DIR = os.path.join(parent_dir, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_or_create_data():
    """
    Load preprocessed data or create a synthetic dataset if data doesn't exist
    """
    data_path = os.path.join(DATA_DIR, "processed_data.npz")
    
    try:
        # Try to load actual data
        print(f"Attempting to load data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))
        
        if np.min(counts) < 2:
            print("Loaded data is too imbalanced. Creating new synthetic data.")
            raise FileNotFoundError
            
        print(f"Data loaded successfully: X shape={X.shape}, y shape={y.shape}")
        return X, y
    except FileNotFoundError:
        # Create synthetic data if no data file exists or if existing data is too imbalanced
        print(f"Creating synthetic data for demonstration.")
        
        # Creating synthetic data - 1000 samples with 10000 features
        n_samples = 1000
        n_features = 10000
        
        np.random.seed(42)
        X_synth = np.random.randn(n_samples, n_features)
        
        # Create synthetic labels (binary classification)
        # Ensure balanced classes
        n_samples_per_class = n_samples // 2
        
        # Generate positive class
        y_synth = np.zeros(n_samples)
        positive_indices = np.random.choice(n_samples, n_samples_per_class, replace=False)
        y_synth[positive_indices] = 1
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1, n_samples)
        y_synth = np.clip(y_synth + noise, 0, 1)
        
        # Ensure we still have balanced classes after adding noise
        y_synth = (y_synth > 0.5).astype(np.float32)
        
        # Verify class balance
        unique, counts = np.unique(y_synth, return_counts=True)
        print("Class distribution in synthetic data:", dict(zip(unique, counts)))
        
        # Save synthetic data
        np.savez(data_path, X=X_synth, y=y_synth)
        print(f"Synthetic data created and saved: X shape={X_synth.shape}, y shape={y_synth.shape}")
        
        return X_synth, y_synth

def preprocess_data(X, y):
    """Preprocess data: normalize features and split into train/test sets"""
    # Verify class balance before splitting
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution before split:", dict(zip(unique, counts)))
    
    # Normalize features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save normalization parameters
    joblib.dump({'scaler': scaler}, os.path.join(MODEL_DIR, "normalization_params.pkl"))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Verify split maintained class balance
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("Train set class distribution:", dict(zip(unique_train, counts_train)))
    print("Test set class distribution:", dict(zip(unique_test, counts_test)))
    
    print(f"Train data: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_train.shape[1]

def binary_accuracy(preds, y):
    """Calculate binary classification accuracy"""
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train_model(model, train_loader, optimizer, criterion, epochs=10):
    """Train a model"""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            acc = binary_accuracy(output, y_batch)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        output = model(X_test)
        acc = binary_accuracy(output, y_test).item()
        
        # ROC-AUC score
        y_pred = output.cpu().numpy()
        y_true = y_test.cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)
        
    return acc, auc

def objective(trial):
    """Objective function for Optuna optimization"""
    global X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim
    
    # Hyperparameters for linear model
    linear_embed_dim = trial.suggest_int("linear_embed_dim", 32, 256)
    linear_lr = trial.suggest_float("linear_lr", 1e-4, 1e-2, log=True)
    linear_weight_decay = trial.suggest_float("linear_weight_decay", 1e-5, 1e-3, log=True)
    
    # Hyperparameters for GBM model
    gbm_embed_dim = trial.suggest_int("gbm_embed_dim", 32, 256)
    gbm_dropout = trial.suggest_float("gbm_dropout", 0.1, 0.5)
    gbm_lr = trial.suggest_float("gbm_lr", 1e-4, 1e-2, log=True)
    gbm_weight_decay = trial.suggest_float("gbm_weight_decay", 1e-5, 1e-3, log=True)
    
    # Hyperparameters for ensemble training
    ensemble_epochs = trial.suggest_int("ensemble_epochs", 5, 20)
    ensemble_lr = trial.suggest_float("ensemble_lr", 1e-3, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Initialize models
    linear_model = LinearRegressionNet(input_dim=input_dim, embed_dim=linear_embed_dim).to(device)
    gbm_model = GBMNet(input_dim=input_dim, embed_dim=gbm_embed_dim).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Train linear model
    linear_optimizer = torch.optim.Adam(
        linear_model.parameters(), 
        lr=linear_lr, 
        weight_decay=linear_weight_decay
    )
    criterion = nn.BCELoss()
    linear_model = train_model(linear_model, train_loader, linear_optimizer, criterion)
    
    # Train GBM model
    gbm_optimizer = torch.optim.Adam(
        gbm_model.parameters(), 
        lr=gbm_lr, 
        weight_decay=gbm_weight_decay
    )
    gbm_model = train_model(gbm_model, train_loader, gbm_optimizer, criterion)
    
    # Create ensemble model
    ensemble = EnsembleModel([linear_model, gbm_model], device).to(device)
    
    # Only optimize ensemble weights (base models are frozen)
    ensemble_optimizer = torch.optim.Adam(
        [ensemble.weights], 
        lr=ensemble_lr
    )
    
    # Train ensemble model
    ensemble = train_model(ensemble, train_loader, ensemble_optimizer, criterion, epochs=ensemble_epochs)
    
    # Evaluate models
    _, linear_auc = evaluate_model(linear_model, X_test_tensor, y_test_tensor)
    _, gbm_auc = evaluate_model(gbm_model, X_test_tensor, y_test_tensor)
    _, ensemble_auc = evaluate_model(ensemble, X_test_tensor, y_test_tensor)
    
    # Log intermediate results
    trial.set_user_attr("linear_auc", float(linear_auc))
    trial.set_user_attr("gbm_auc", float(gbm_auc))
    
    # Return ensemble AUC (to maximize)
    return ensemble_auc

def save_optimization_plots(study):
    """Save optimization visualization plots"""
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "optimization_history.png"))
    
    # Plot parameter importances
    plt.figure(figsize=(10, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "parameter_importances.png"))

def save_best_model(study):
    """Save the best model from optimization"""
    global X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim
    
    # Get best parameters
    params = study.best_params
    print("Best parameters:", params)
    
    # Recreate the best model
    linear_model = LinearRegressionNet(input_dim=input_dim, embed_dim=params["linear_embed_dim"]).to(device)
    gbm_model = GBMNet(input_dim=input_dim, embed_dim=params["gbm_embed_dim"]).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    
    # Train linear model
    linear_optimizer = torch.optim.Adam(
        linear_model.parameters(), 
        lr=params["linear_lr"], 
        weight_decay=params["linear_weight_decay"]
    )
    criterion = nn.BCELoss()
    linear_model = train_model(linear_model, train_loader, linear_optimizer, criterion)
    
    # Train GBM model
    gbm_optimizer = torch.optim.Adam(
        gbm_model.parameters(), 
        lr=params["gbm_lr"], 
        weight_decay=params["gbm_weight_decay"]
    )
    gbm_model = train_model(gbm_model, train_loader, gbm_optimizer, criterion)
    
    # Create ensemble model
    ensemble = EnsembleModel([linear_model, gbm_model], device).to(device)
    
    # Only optimize ensemble weights
    ensemble_optimizer = torch.optim.Adam(
        [ensemble.weights], 
        lr=params["ensemble_lr"]
    )
    
    # Train ensemble model
    ensemble = train_model(ensemble, train_loader, ensemble_optimizer, criterion, epochs=params["ensemble_epochs"])
    
    # Evaluate final model
    acc, auc = evaluate_model(ensemble, X_test_tensor, y_test_tensor)
    
    print(f"Final model: Accuracy={acc:.4f}, AUC={auc:.4f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "linear_gbm_ensemble_balanced_optimized.pth")
    
    # Create checkpoint with model state and architecture information
    checkpoint = {
        'state_dict': ensemble.state_dict(),
        'architecture': {
            'input_dim': input_dim,
            'embed_dim': max(params["linear_embed_dim"], params["gbm_embed_dim"]),
            'models': ['linear', 'gbm']  # Model types in the ensemble
        },
        'hyperparameters': params,
        'performance': {
            'accuracy': acc,
            'auc': auc
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save the checkpoint
    torch.save(checkpoint, model_path)
    print(f"Best model saved to {model_path}")
    
    # Also save a default named version for the Flask app to easily find
    torch.save(checkpoint, os.path.join(MODEL_DIR, "linear_gbm_ensemble_balanced.pth"))
    print(f"Default model saved to {os.path.join(MODEL_DIR, 'linear_gbm_ensemble_balanced.pth')}")

def main():
    # Load data
    global X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim
    
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Load or create data
        X, y = load_or_create_data()
        
        # Preprocess data
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim = preprocess_data(X, y)
        
        # Get number of trials from environment variable or use default
        n_trials = int(os.environ.get("N_TRIALS", 50))
        study_name = os.environ.get("STUDY_NAME", "ensemble_optimization")
        
        print(f"Starting optimization with {n_trials} trials")
        
        # Create study
        storage = f"sqlite:///{study_name}.db"
        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Print statistics
        pruned_trials = study.get_trials(states=[TrialState.PRUNED])
        complete_trials = study.get_trials(states=[TrialState.COMPLETE])
        
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        
        print("Best trial:")
        trial = study.best_trial
        
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        # Save optimization plots
        save_optimization_plots(study)
        
        # Save best model
        save_best_model(study)
        
        return study
        
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
