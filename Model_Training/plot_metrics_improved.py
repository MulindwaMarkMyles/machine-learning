import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json

def plot_confusion_matrices(teacher_preds, student_preds, true_labels, save_path):
    """Plot confusion matrices for teacher and student models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Teacher confusion matrix
    cm_teacher = confusion_matrix(true_labels, teacher_preds)
    sns.heatmap(cm_teacher, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Teacher Model\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Student confusion matrix
    cm_student = confusion_matrix(true_labels, student_preds)
    sns.heatmap(cm_student, annot=True, fmt='d', ax=ax2, cmap='Greens')
    ax2.set_title('Student Model\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_progression(metrics, save_path):
    """Plot the progression of student learning compared to teacher"""
    epochs = range(1, len(metrics['train_f1']) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot F1 scores
    plt.plot(epochs, metrics['teacher_train_f1'], 'b-', label='Teacher Train F1')
    plt.plot(epochs, metrics['teacher_val_f1'], 'b--', label='Teacher Val F1')
    plt.plot(epochs, metrics['train_f1'], 'g-', label='Student Train F1')
    plt.plot(epochs, metrics['val_f1'], 'g--', label='Student Val F1')
    
    plt.title('Knowledge Distillation Progress')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_performance_gap(metrics, save_path):
    """Plot the gap between teacher and student performance"""
    epochs = range(1, len(metrics['teacher_student_diff']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['teacher_student_diff'], 'r-')
    plt.title('Teacher-Student Performance Gap')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal line for initial gap
    plt.axhline(y=metrics['teacher_student_diff'][0], 
               color='k', linestyle='--', 
               label=f"Initial Gap: {metrics['teacher_student_diff'][0]:.4f}")
    
    # Add horizontal line for final gap
    plt.axhline(y=metrics['teacher_student_diff'][-1], 
               color='g', linestyle='--', 
               label=f"Final Gap: {metrics['teacher_student_diff'][-1]:.4f}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(metrics, save_path):
    """Plot comparison of teacher and student model metrics"""
    comparison = {
        'Accuracy': [metrics['teacher_accuracy'], metrics['student_accuracy']],
        'F1 Score': [metrics['teacher_f1'], metrics['student_f1']],
        'Precision': [metrics['teacher_precision'], metrics['student_precision']],
        'Recall': [metrics['teacher_recall'], metrics['student_recall']]
    }
    
    df = pd.DataFrame(comparison, index=['Teacher', 'Student'])
    
    plt.figure(figsize=(10, 6))
    ax = df.plot(kind='bar', figsize=(12, 6), rot=0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
        
    plt.title('Teacher vs Student Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metrics')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        # Try to load the detailed metrics
        with open('./models/distillation_metrics.json', 'r') as f:
            data = json.load(f)
        
        print("Successfully loaded distillation metrics")
        
        # Create all plots
        print("Generating confusion matrices...")
        plot_confusion_matrices(
            data['teacher_predictions'],
            data['student_predictions'],
            data['true_labels'],
            './models/confusion_matrices.png'
        )
        
        print("Generating training progression plot...")
        plot_training_progression(
            data['training_history'],
            './models/training_progression.png'
        )
        
        print("Generating performance gap plot...")
        plot_performance_gap(
            data['training_history'],
            './models/performance_gap.png'
        )
        
        print("Generating model comparison plot...")
        plot_model_comparison(
            data,
            './models/model_comparison.png'
        )
        
        print("\nAll plots generated successfully!")
        print(f"Initial Gap: {data['initial_gap']:.4f}")
        print(f"Final Gap: {data['final_gap']:.4f}")
        print(f"Gap Reduction: {data['gap_reduction']:.4f}")
        
    except FileNotFoundError:
        print("Error: distillation_metrics.json not found!")
        print("Please run model_distillation.py first to generate metrics.")
    except Exception as e:
        print(f"Error generating plots: {str(e)}")

if __name__ == "__main__":
    main()
