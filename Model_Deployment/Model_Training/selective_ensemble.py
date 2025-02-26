import torch
import torch.nn as nn
import json
from multimodel_train import (MATDataset, LinearRegressionNet, LogisticRegressionNet,
                            TreeBasedNet, SVMNet, KNNNet, GBMNet, XGBoostNet, 
                            DeepNeuralNet, PCANet)
from torch.utils.data import DataLoader
import joblib
import numpy as np
from sklearn.metrics import precision_score

class SelectiveEnsemble(nn.Module):
    def __init__(self, models, device, voting='weighted'):
        super(SelectiveEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        self.device = device
        self.voting = voting

    def forward(self, x):
        if self.voting == 'weighted':
            predictions = torch.zeros(x.size(0), 1).to(self.device)
            weights = torch.softmax(self.weights, dim=0)
            for i, model in enumerate(self.models):
                predictions += weights[i] * model(x)
            return torch.sigmoid(predictions)
        else:  # majority voting
            predictions = torch.zeros(x.size(0), len(self.models)).to(self.device)
            for i, model in enumerate(self.models):
                predictions[:, i] = model(x).squeeze()
            return (predictions.mean(dim=1, keepdim=True) > 0.5).float()

def evaluate_model_precision(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predicted = (outputs.squeeze() > 0.5).int()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return precision_score(all_labels, all_preds)

def select_best_models(models, model_names, val_loader, device, top_k=3):
    precisions = []
    for model in models:
        precision = evaluate_model_precision(model, val_loader, device)
        precisions.append(precision)
    
    # Get indices of top-k models
    top_indices = np.argsort(precisions)[-top_k:]
    selected_models = [models[i] for i in top_indices]
    selected_names = [model_names[i] for i in top_indices]
    selected_precisions = [precisions[i] for i in top_indices]
    
    return selected_models, selected_names, selected_precisions

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    # Load fine-tuned results to get model performances
    with open('./models/fine_tuned_results.json', 'r') as f:
        fine_tuned_results = json.load(f)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load all models
    input_dim = dataset[0][0].shape[1] * dataset[0][0].shape[0]
    models = []
    model_names = ['linear', 'logistic', 'tree', 'svm', 'knn', 'gbm', 'xgboost', 'neural', 'pca']
    
    for model_name in model_names:
        config_path = f'./models/{model_name}_config.pkl'
        weights_path = f'./models/{model_name}_best.pth'
        model = load_model(model_name, input_dim, config_path, weights_path)  # Using the same load_model from ensemble_all.py
        model.to(device)
        model.eval()
        models.append(model)
    
    # Select best models based on precision
    selected_models, selected_names, precisions = select_best_models(
        models, model_names, val_loader, device, top_k=3
    )
    
    print("\nSelected Models:")
    for name, precision in zip(selected_names, precisions):
        print(f"{name}: Precision = {precision:.4f}")
    
    # Create and train selective ensemble
    ensemble = SelectiveEnsemble(selected_models, device, voting='weighted').to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam([
        {'params': ensemble.models.parameters(), 'lr': 1e-4},
        {'params': ensemble.weights, 'lr': 1e-2}
    ])
    
    # Training loop
    best_precision = 0
    for epoch in range(10):
        ensemble.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = ensemble(data)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            # Ensure weights stay positive and sum to 1
            with torch.no_grad():
                ensemble.weights.data = torch.softmax(ensemble.weights.data, dim=0)
        
        # Evaluate ensemble precision
        precision = evaluate_model_precision(ensemble, val_loader, device)
        if precision > best_precision:
            best_precision = precision
            torch.save(ensemble.state_dict(), './models/selective_ensemble_best.pth')
        
        print(f'Epoch {epoch+1}, Validation Precision: {precision:.4f}')
    
    print(f"\nBest Selective Ensemble Precision: {best_precision:.4f}")
    print("\nFinal model weights:")
    for name, weight in zip(selected_names, ensemble.weights.data):
        print(f"{name}: {weight.item():.4f}")

if __name__ == "__main__":
    main()
