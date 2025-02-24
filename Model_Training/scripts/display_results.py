import json
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def create_performance_table(results):
    # Prepare data for tabulation
    table_data = []
    headers = ['Model', 'Original Score', 'Fine-tuned Score', 'Improvement', 
              'Best Score', 'Batch Size', 'Learning Rate', 'Dropout']

    for model_name, model_data in results.items():
        row = [
            model_name.upper(),
            f"{model_data['original_score']:.4f}",
            f"{model_data['fine_tuned_score']:.4f}",
            f"{model_data['improvement']:+.4f}",
            f"{model_data['fine_tuned_score']:.4f}",
            model_data['best_params']['batch_size'],
            f"{model_data['best_params']['lr']:.2e}",
            f"{model_data['best_params']['dropout']:.2f}"
        ]
        table_data.append(row)

    # Sort by fine-tuned score
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    return table_data, headers

def plot_comparison(results):
    models = list(results.keys())
    original_scores = [results[m]['original_score'] for m in models]
    fine_tuned_scores = [results[m]['fine_tuned_score'] for m in models]
    improvements = [results[m]['improvement'] for m in models]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Bar plot comparing original and fine-tuned scores
    x = range(len(models))
    width = 0.35
    ax1.bar([i - width/2 for i in x], original_scores, width, label='Original', color='lightblue')
    ax1.bar([i + width/2 for i in x], fine_tuned_scores, width, label='Fine-tuned', color='darkblue')
    ax1.set_ylabel('Score')
    ax1.set_title('Original vs Fine-tuned Scores by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar plot showing improvements
    ax2.bar(models, improvements, color=['g' if i > 0 else 'r' for i in improvements])
    ax2.set_ylabel('Improvement')
    ax2.set_title('Score Improvement by Model')
    ax2.set_xticklabels([m.upper() for m in models], rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    # Load results
    results = load_results('../models/fine_tuned_results.json')

    # Create and display performance table
    table_data, headers = create_performance_table(results)
    print("\nModel Performance Summary:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    improvements = [float(row[3]) for row in table_data]
    print(f"Average improvement: {sum(improvements)/len(improvements):+.4f}")
    print(f"Best improvement: {max(improvements):+.4f} ({table_data[0][0]} model)")
    print(f"Number of models improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    # Create and save visualization
    fig = plot_comparison(results)
    plt.savefig('../models/performance_comparison.png')
    plt.close()

    print("\nVisualization has been saved to '../models/performance_comparison.png'")

if __name__ == "__main__":
    main()
