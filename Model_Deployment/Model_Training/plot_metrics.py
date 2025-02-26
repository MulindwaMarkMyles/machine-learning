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
    sns.heatmap(cm_teacher, annot=True, fmt='d', ax=ax1)
    ax1.set_title('Teacher Model\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Student confusion matrix
    cm_student = confusion_matrix(true_labels, student_preds)
    sns.heatmap(cm_student, annot=True, fmt='d', ax=ax2)
    ax2.set_title('Student Model\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(metrics, save_path):
    """Plot comparison of teacher and student model metrics"""
    metrics_df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(metrics_df.columns))
    
    plt.bar(index, metrics_df.loc['Teacher'], bar_width, label='Teacher')
    plt.bar(index + bar_width, metrics_df.loc['Student'], bar_width, label='Student')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Teacher vs Student Model Performance')
    plt.xticks(index + bar_width/2, metrics_df.columns)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load metrics from distillation training
    with open('./models/distillation_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Plot confusion matrices
    plot_confusion_matrices(
        metrics['teacher_predictions'],
        metrics['student_predictions'],
        metrics['true_labels'],
        './models/confusion_matrices.png'
    )
    
    # Plot model comparison
    comparison_metrics = {
        'Accuracy': [metrics['teacher_accuracy'], metrics['student_accuracy']],
        'F1 Score': [metrics['teacher_f1'], metrics['student_f1']],
        'Precision': [metrics['teacher_precision'], metrics['student_precision']],
        'Recall': [metrics['teacher_recall'], metrics['student_recall']]
    }
    
    plot_model_comparison(
        pd.DataFrame(comparison_metrics, index=['Teacher', 'Student']),
        './models/model_comparison.png'
    )

if __name__ == "__main__":
    main()
