import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from multimodel_train import MATDataset
from ensemble_all import load_model  # Import the load_model function

class DistillationStudent(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class ContrastiveLearner(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        self.temperature = 0.07
        
    def forward(self, anchor, positive, negative):
        # Get embeddings
        anchor_emb = self.teacher.get_embedding(anchor)
        positive_emb = self.teacher.get_embedding(positive)
        negative_emb = self.teacher.get_embedding(negative)
        
        # Compute similarities
        pos_sim = torch.cosine_similarity(anchor_emb, positive_emb, dim=1)
        neg_sim = torch.cosine_similarity(anchor_emb, negative_emb, dim=1)
        
        # Compute loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return nn.CrossEntropyLoss()(logits, labels)

def select_teacher_student_models(results_path, model_names):
    """Select teacher and student models based on performance"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract model performances from combinations
    model_performances = {}
    for combo in results['combinations']:
        for model in combo['models']:
            if model not in model_performances and model in model_names:
                model_performances[model] = combo['metrics']['precision']
    
    # Convert to list and sort by precision
    sorted_models = sorted(
        [(model, precision) for model, precision in model_performances.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    if not sorted_models:
        raise ValueError("No valid models found in results file")
    
    print("\nModel performances:")
    for model, precision in sorted_models:
        print(f"{model}: {precision:.4f}")
    
    # Select best model as teacher
    teacher_model = sorted_models[0][0]
    teacher_precision = sorted_models[0][1]
    
    # Find a model with precision around 80% of the best model
    target_precision = teacher_precision * 0.8
    student_model = None
    
    for model, precision in sorted_models[1:]:
        if precision <= target_precision:
            student_model = model
            break
    
    # If no model found with desired precision gap, take the second best model
    if student_model is None:
        student_model = sorted_models[1][0] if len(sorted_models) > 1 else sorted_models[0][0]
    
    print(f"\nSelected teacher model: {teacher_model} (precision: {teacher_precision:.4f})")
    print(f"Selected student model: {student_model} (precision: {model_performances[student_model]:.4f})")
    
    return teacher_model, student_model

def train_knowledge_distillation(teacher_model, student_model, train_loader, 
                               val_loader, device, num_epochs=50):
    """Train student model using knowledge distillation from teacher"""
    temperature = 4.0  # Increased temperature for softer probabilities
    alpha = 0.7  # Higher weight on distillation loss
    
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    criterion_task = nn.BCELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=5e-4)  # Increased learning rate
    
    # Collect all evaluation data
    all_metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'teacher_student_diff': [],
        'teacher_train_f1': [],  # Track teacher performance for comparison
        'teacher_val_f1': []
    }
    
    # Initial evaluation of teacher model
    teacher_train_metrics = evaluate_model(teacher_model, train_loader, device)
    teacher_val_metrics = evaluate_model(teacher_model, val_loader, device)
    
    print(f"\nInitial Teacher Performance:")
    print(f"Train F1: {teacher_train_metrics['f1']:.4f}, Val F1: {teacher_val_metrics['f1']:.4f}")
    
    # Initial evaluation of student model
    student_train_metrics = evaluate_model(student_model, train_loader, device)
    student_val_metrics = evaluate_model(student_model, val_loader, device)
    
    print(f"Initial Student Performance:")
    print(f"Train F1: {student_train_metrics['f1']:.4f}, Val F1: {student_val_metrics['f1']:.4f}")
    
    # Store initial performance gap
    initial_train_gap = abs(teacher_train_metrics['f1'] - student_train_metrics['f1'])
    initial_val_gap = abs(teacher_val_metrics['f1'] - student_val_metrics['f1'])
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        # Training
        student_model.train()
        teacher_model.eval()
        train_loss = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Get predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            
            student_outputs = student_model(data)
            
            # Reshape outputs if needed for KLDivLoss
            student_logits = torch.cat([1-student_outputs, student_outputs], dim=1)
            teacher_logits = torch.cat([1-teacher_outputs, teacher_outputs], dim=1)
            
            # Compute losses
            distillation_loss = criterion_kd(
                torch.log_softmax(student_logits/temperature, dim=1),
                torch.softmax(teacher_logits/temperature, dim=1)
            ) * (temperature ** 2)
            
            task_loss = criterion_task(student_outputs.squeeze(), labels.float())
            loss = alpha * distillation_loss + (1 - alpha) * task_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate both models
        student_train_metrics = evaluate_model(student_model, train_loader, device)
        student_val_metrics = evaluate_model(student_model, val_loader, device)
        
        # Only evaluate teacher once per epoch to save time
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            teacher_train_metrics = evaluate_model(teacher_model, train_loader, device)
            teacher_val_metrics = evaluate_model(teacher_model, val_loader, device)
        
        # Calculate performance gaps
        train_gap = abs(teacher_train_metrics['f1'] - student_train_metrics['f1'])
        val_gap = abs(teacher_val_metrics['f1'] - student_val_metrics['f1'])
        
        # Store metrics
        all_metrics['train_loss'].append(train_loss / len(train_loader))
        all_metrics['val_loss'].append(student_val_metrics['loss'])
        all_metrics['train_acc'].append(student_train_metrics['accuracy'])
        all_metrics['val_acc'].append(student_val_metrics['accuracy'])
        all_metrics['train_f1'].append(student_train_metrics['f1'])
        all_metrics['val_f1'].append(student_val_metrics['f1'])
        all_metrics['teacher_student_diff'].append(val_gap)
        all_metrics['teacher_train_f1'].append(teacher_train_metrics['f1'])
        all_metrics['teacher_val_f1'].append(teacher_val_metrics['f1'])
        
        # Print progress every few epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Teacher F1: {teacher_val_metrics['f1']:.4f}, Student F1: {student_val_metrics['f1']:.4f}")
            print(f"Performance Gap: {val_gap:.4f}, Loss: {train_loss/len(train_loader):.4f}")
    
    # Collect final predictions for confusion matrices
    teacher_preds, student_preds, true_labels = get_all_predictions(
        teacher_model, student_model, val_loader, device
    )
    
    # Create detailed metrics for plotting
    detailed_metrics = {
        'training_history': all_metrics,
        'teacher_predictions': teacher_preds.tolist(),
        'student_predictions': student_preds.tolist(),
        'true_labels': true_labels.tolist(),
        'teacher_accuracy': teacher_val_metrics['accuracy'],
        'student_accuracy': student_val_metrics['accuracy'],
        'teacher_f1': teacher_val_metrics['f1'],
        'student_f1': student_val_metrics['f1'],
        'teacher_precision': teacher_val_metrics['precision'],
        'student_precision': student_val_metrics['precision'],
        'teacher_recall': teacher_val_metrics['recall'], 
        'student_recall': student_val_metrics['recall'],
        'initial_gap': initial_val_gap,
        'final_gap': val_gap,
        'gap_reduction': initial_val_gap - val_gap
    }
    
    return all_metrics, detailed_metrics

def evaluate_model(model, data_loader, device):
    """Evaluate a model and return metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels.float())
            
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).int()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_all_predictions(teacher_model, student_model, data_loader, device):
    """Get all predictions from both models for comparison"""
    teacher_model.eval()
    student_model.eval()
    all_teacher_preds = []
    all_student_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            
            teacher_outputs = teacher_model(data)
            teacher_preds = (teacher_outputs.squeeze() > 0.5).int()
            
            student_outputs = student_model(data)
            student_preds = (student_outputs.squeeze() > 0.5).int()
            
            all_teacher_preds.extend(teacher_preds.cpu().numpy())
            all_student_preds.extend(student_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_teacher_preds), np.array(all_student_preds), np.array(all_labels)

def calculate_f1(predictions, labels):
    """Calculate F1 score"""
    from sklearn.metrics import f1_score
    return f1_score(labels, predictions)

def plot_training_curves(metrics_history, save_path):
    """Plot training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(metrics_history['train_loss'], label='Train Loss')
    ax1.plot(metrics_history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy curves
    ax2.plot(metrics_history['train_acc'], label='Train Accuracy')
    ax2.plot(metrics_history['val_acc'], label='Val Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # F1 Score curves
    ax3.plot(metrics_history['train_f1'], label='Train F1')
    ax3.plot(metrics_history['val_f1'], label='Val F1')
    ax3.set_title('F1 Score Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    adhd_folder = "../ADHD_part2/ADHD_part2"
    control_folder = "../Control_part2/Control_part2"
    dataset = MATDataset(adhd_folder, control_folder)
    
    # Setup data loaders
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Select teacher and student models
    results_path = './models/ensemble_combinations_results.json'
    model_names = ['linear', 'logistic', 'tree', 'svm', 'knn', 'gbm', 'xgboost', 'neural', 'pca']
    teacher_name, student_name = select_teacher_student_models(results_path, model_names)
    
    print(f"Selected teacher model: {teacher_name}")
    print(f"Selected student model: {student_name}")
    
    # Load models
    input_dim = dataset[0][0].shape[1] * dataset[0][0].shape[0]
    
    teacher_model = load_model(
        teacher_name, 
        input_dim,
        f'./models/{teacher_name}_config.pkl',
        f'./models/{teacher_name}_best.pth'
    ).to(device)
    
    student_model = load_model(
        student_name,
        input_dim,
        f'./models/{student_name}_config.pkl',
        f'./models/{student_name}_best.pth'
    ).to(device)
    
    # Train with knowledge distillation
    print(f"\nStarting knowledge distillation from {teacher_name} to {student_name}...")
    metrics_history, detailed_metrics = train_knowledge_distillation(
        teacher_model, student_model,
        train_loader, val_loader,
        device, num_epochs=30  # Reduced epochs for faster training
    )
    
    # Plot training curves
    plot_training_curves(metrics_history, './models/distillation_training_curves.png')
    
    # Save detailed metrics for plotting
    with open('./models/distillation_metrics.json', 'w') as f:
        json.dump(detailed_metrics, f, indent=4)
    
    # Save improved student model
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'original_model_name': student_name,
        'teacher_model_name': teacher_name,
        'training_history': metrics_history
    }, f'./models/{student_name}_distilled.pth')
    
    # Print final metrics
    print("\nFinal Metrics:")
    print("-" * 50)
    print(f"Teacher Model: {teacher_name}")
    print(f"Student Model: {student_name}")
    print(f"Final Validation F1: {metrics_history['val_f1'][-1]:.4f}")
    
    initial_diff = metrics_history['teacher_student_diff'][0]
    final_diff = metrics_history['teacher_student_diff'][-1]
    improvement = initial_diff - final_diff
    
    print(f"Initial Performance Gap: {initial_diff:.4f}")
    print(f"Final Performance Gap: {final_diff:.4f}")
    print(f"Gap Reduction: {improvement:.4f}")
    
    print("\nDetailed metrics saved to ./models/distillation_metrics.json")

if __name__ == "__main__":
    main()
