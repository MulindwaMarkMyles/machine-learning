import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from multimodel_train import (MATDataset, LinearRegressionNet, LogisticRegressionNet,
                            TreeBasedNet, SVMNet, KNNNet, GBMNet, XGBoostNet, 
                            DeepNeuralNet, PCANet, validate)

class HyperparameterOptimizer:
    def __init__(self, dataset, device, model_name, n_trials=5):
        self.dataset = dataset
        self.device = device
        self.model_name = model_name
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = 0
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        self.train_data, self.val_data, self.test_data = random_split(
            dataset, [train_size, val_size, test_size])

    def objective(self, trial):
        # Common hyperparameters
        batch_size = trial.suggest_int('batch_size', 16, 128)
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        embed_dim = trial.suggest_int('embed_dim', 64, 512, step=64)
        
        # Model specific hyperparameters
        if self.model_name == 'pca':
            n_components = trial.suggest_int('n_components', 32, 256, step=32)
            model = PCANet(
                input_dim=self.dataset[0][0].shape[1] * self.dataset[0][0].shape[0],
                embed_dim=embed_dim,
                n_components=n_components
            )
        else:
            model_classes = {
                'linear': LinearRegressionNet,
                'logistic': LogisticRegressionNet,
                'tree': TreeBasedNet,
                'svm': SVMNet,
                'knn': KNNNet,
                'gbm': GBMNet,
                'xgboost': XGBoostNet,
                'neural': DeepNeuralNet
            }
            
            model = model_classes[self.model_name](
                input_dim=self.dataset[0][0].shape[1] * self.dataset[0][0].shape[0],
                embed_dim=embed_dim
            )

        model = model.to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), 
                              lr=lr,
                              weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True))
        
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
        
        # Training loop
        n_epochs = trial.suggest_int('n_epochs', 10, 50)
        best_val_acc = 0
        
        for epoch in range(n_epochs):
            model.train()
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                             trial.suggest_float('clip_value', 0.1, 5.0))
                optimizer.step()
            
            # Validation
            val_loss, val_acc = validate(model, val_loader, criterion, self.device)
            
            # Pruning (early stopping for unpromising trials)
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            best_val_acc = max(best_val_acc, val_acc)
        
        return best_val_acc

    def optimize(self):
        study = optuna.create_study(direction='maximize',
                                  pruner=optuna.pruners.MedianPruner())
        study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return self.best_params, self.best_score

    def train_with_best_params(self):
        if self.best_params is None:
            raise ValueError("Run optimize() first to find the best parameters")
        
        # Create model with best parameters
        if self.model_name == 'pca':
            model = PCANet(
                input_dim=self.dataset[0][0].shape[1] * self.dataset[0][0].shape[0],
                embed_dim=self.best_params['embed_dim'],
                n_components=self.best_params['n_components']
            )
        else:
            model_classes = {
                'linear': LinearRegressionNet,
                'logistic': LogisticRegressionNet,
                'tree': TreeBasedNet,
                'svm': SVMNet,
                'knn': KNNNet,
                'gbm': GBMNet,
                'xgboost': XGBoostNet,
                'neural': DeepNeuralNet
            }
            
            model = model_classes[self.model_name](
                input_dim=self.dataset[0][0].shape[1] * self.dataset[0][0].shape[0],
                embed_dim=self.best_params['embed_dim']
            )

        model = model.to(self.device)
        
        # Train with best parameters
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), 
                              lr=self.best_params['lr'],
                              weight_decay=self.best_params['weight_decay'])
        
        train_loader = DataLoader(self.train_data, 
                                batch_size=self.best_params['batch_size'],
                                shuffle=True)
        
        # Final training
        for epoch in range(self.best_params['n_epochs']):
            model.train()
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                             self.best_params['clip_value'])
                optimizer.step()
        
        # Save the optimized model
        torch.save(model.state_dict(), 
                  f'./models/{self.model_name}_optimized.pth')
        return model

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    # List of models to optimize
    models_to_optimize = [
        'linear', 'logistic', 'tree', 'svm', 'knn', 
        'gbm', 'xgboost', 'neural', 'pca'
    ]
    
    # Run optimization for each model
    results = {}
    for model_name in models_to_optimize:
        print(f"\nOptimizing {model_name} model...")
        optimizer = HyperparameterOptimizer(dataset, device, model_name)
        best_params, best_score = optimizer.optimize()
        
        # Train final model with best parameters
        final_model = optimizer.train_with_best_params()
        
        results[model_name] = {
            'best_params': best_params,
            'best_score': best_score
        }
        
        print(f"\nBest parameters for {model_name}:")
        print(best_params)
        print(f"Best validation accuracy: {best_score:.4f}")
    
    # Save optimization results
    import json
    with open('./models/optimization_results.json', 'w') as f:
        json.dump(results, f, indent=4)
