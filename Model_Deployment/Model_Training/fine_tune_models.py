import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
from multimodel_train import (MATDataset, LinearRegressionNet, LogisticRegressionNet,
                            TreeBasedNet, SVMNet, KNNNet, GBMNet, XGBoostNet, 
                            DeepNeuralNet, PCANet, validate)

class ModelFineTuner:
    def __init__(self, dataset, device, model_name, base_params, n_trials=5):
        self.dataset = dataset
        self.device = device
        self.model_name = model_name
        self.base_params = base_params
        self.n_trials = n_trials
        
        # Split dataset with more emphasis on validation
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        self.train_data, self.val_data, self.test_data = random_split(
            dataset, [train_size, val_size, test_size])
        
    def create_search_space(self, trial):
        """Create a narrow search space around the best parameters"""
        base = self.base_params
        
        # Define search ranges as percentages around the base values
        params = {
            'batch_size': trial.suggest_int('batch_size', 
                max(8, int(base['batch_size'] * 0.8)),
                min(256, int(base['batch_size'] * 1.2))),
            
            'lr': trial.suggest_float('lr', 
                base['lr'] * 0.5,
                base['lr'] * 1.5,
                log=True),
            
            'dropout': trial.suggest_float('dropout',
                max(0.1, base['dropout'] - 0.1),
                min(0.5, base['dropout'] + 0.1)),
            
            'embed_dim': trial.suggest_int('embed_dim',
                max(32, int(base['embed_dim'] * 0.8)),
                min(768, int(base['embed_dim'] * 1.2)),
                step=32),
            
            'weight_decay': trial.suggest_float('weight_decay',
                base['weight_decay'] * 0.5,
                base['weight_decay'] * 1.5,
                log=True),
            
            'n_epochs': trial.suggest_int('n_epochs',
                max(5, base['n_epochs'] - 5),
                base['n_epochs'] + 5),
            
            'clip_value': trial.suggest_float('clip_value',
                max(0.1, base['clip_value'] * 0.8),
                base['clip_value'] * 1.2)
        }
        
        # Add PCA-specific parameter if applicable
        if self.model_name == 'pca' and 'n_components' in base:
            params['n_components'] = trial.suggest_int('n_components',
                max(16, int(base['n_components'] * 0.8)),
                min(512, int(base['n_components'] * 1.2)),
                step=16)
        
        return params

    def create_model(self, input_dim):
        """Create and return the appropriate model based on model_name"""
        params = self.base_params
        
        if self.model_name == 'pca':
            model = PCANet(input_dim, params['embed_dim'], params['n_components'])
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
            model = model_classes[self.model_name](input_dim, params['embed_dim'])
        
        model = model.to(self.device)
        return model

    def objective(self, trial):
        params = self.create_search_space(trial)
        
        # Create model
        input_dim = self.dataset[0][0].shape[1] * self.dataset[0][0].shape[0]
        
        model = self.create_model(input_dim)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), 
                              lr=params['lr'],
                              weight_decay=params['weight_decay'])
        
        train_loader = DataLoader(self.train_data, 
                                batch_size=params['batch_size'],
                                shuffle=True)
        val_loader = DataLoader(self.val_data,
                              batch_size=params['batch_size'],
                              shuffle=False)
        
        # Training with early stopping
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(params['n_epochs']):
            model.train()
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_value'])
                optimizer.step()
            
            val_loss, val_acc = validate(model, val_loader, criterion, self.device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_acc

    def save_model(self, model, model_name, params):
        """Save model state and architecture parameters"""
        save_dict = {
            'state_dict': model.state_dict(),
            'architecture': {
                'input_dim': model.features[0].in_features,
                'embed_dim': model.features[0].out_features,
                'n_components': params.get('n_components', None)
            }
        }
        torch.save(save_dict, f'./models/{model_name}_optimized.pth')

    def fine_tune(self):
        study = optuna.create_study(direction='maximize',
                                  pruner=optuna.pruners.MedianPruner())
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save best model
        input_dim = self.dataset[0][0].shape[1] * self.dataset[0][0].shape[0]
        best_model = self.create_model(input_dim)
        self.save_model(best_model, self.model_name, study.best_params)
        
        return study.best_params, study.best_value

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    # Load previous optimization results
    with open('./models/optimization_results.json', 'r') as f:
        previous_results = json.load(f)
    
    # Fine-tune each model
    fine_tuned_results = {}
    
    for model_name, model_data in previous_results.items():
        print(f"\nFine-tuning {model_name} model...")
        
        tuner = ModelFineTuner(
            dataset=dataset,
            device=device,
            model_name=model_name,
            base_params=model_data['best_params']
        )
        
        best_params, best_score = tuner.fine_tune()
        
        improvement = best_score - model_data['best_score']
        
        fine_tuned_results[model_name] = {
            'original_score': model_data['best_score'],
            'fine_tuned_score': best_score,
            'improvement': improvement,
            'best_params': best_params
        }
        
        print(f"Original score: {model_data['best_score']:.4f}")
        print(f"Fine-tuned score: {best_score:.4f}")
        print(f"Improvement: {improvement:.4f}")
        
    # Save fine-tuned results
    with open('./models/fine_tuned_results.json', 'w') as f:
        json.dump(fine_tuned_results, f, indent=4)
    
    # Print summary
    print("\nFine-tuning Summary:")
    print("-" * 50)
    for model_name, results in fine_tuned_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Original Score: {results['original_score']:.4f}")
        print(f"Fine-tuned Score: {results['fine_tuned_score']:.4f}")
        print(f"Improvement: {results['improvement']:.4f}")
