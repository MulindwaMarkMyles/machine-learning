import json
from tabulate import tabulate
import numpy as np

def load_and_format_results(json_path):
    # Load results
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Prepare table data
    table_data = []
    for combo in data['combinations']:
        row = {
            'Models': ' + '.join(combo['models']),
            'Weights': ' | '.join([f'{w:.3f}' for w in combo['weights']]),
            'Accuracy': f"{combo['metrics']['accuracy']:.4f}",
            'Precision': f"{combo['metrics']['precision']:.4f}",
            'Recall': f"{combo['metrics']['recall']:.4f}",
            'F1': f"{combo['metrics']['f1']:.4f}",
            'Specificity': f"{combo['metrics']['specificity']:.4f}"
        }
        table_data.append(row)
    
    return table_data

def print_top_combinations(table_data, metric='F1', top_n=10):
    # Sort by specified metric
    sorted_data = sorted(table_data, 
                        key=lambda x: float(x[metric]), 
                        reverse=True)[:top_n]
    
    # Print table
    headers = {
        'Models': 'Model Combination',
        'Weights': 'Model Weights',
        'Accuracy': 'Accuracy',
        'Precision': 'Precision',
        'Recall': 'Recall',
        'F1': 'F1 Score',
        'Specificity': 'Specificity'
    }
    
    print(f"\nTop {top_n} Model Combinations by {metric}")
    print("=" * 100)
    print(tabulate(sorted_data, 
                  headers=headers, 
                  tablefmt='grid',
                  numalign='center',
                  stralign='center'))

def print_metric_summary(table_data):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    summary_data = []
    
    for metric in metrics:
        values = [float(row[metric]) for row in table_data]
        summary_row = {
            'Metric': metric,
            'Mean': f"{np.mean(values):.4f}",
            'Std': f"{np.std(values):.4f}",
            'Min': f"{np.min(values):.4f}",
            'Max': f"{np.max(values):.4f}"
        }
        summary_data.append(summary_row)
    
    print("\nMetric Summary Statistics")
    print("=" * 80)
    print(tabulate(summary_data,
                  headers='keys',
                  tablefmt='grid',
                  numalign='center',
                  stralign='center'))

def main():
    json_path = './models/ensemble_combinations_results.json'
    table_data = load_and_format_results(json_path)
    
    # Print top combinations for each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    for metric in metrics:
        print_top_combinations(table_data, metric=metric, top_n=5)
    
    # Print summary statistics
    print_metric_summary(table_data)

if __name__ == "__main__":
    main()
