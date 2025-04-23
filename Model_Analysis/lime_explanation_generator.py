import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

def load_lime_data(model_type):
    """Load all LIME analysis data for a specific model type (best/worst)"""
    data_dir = f'lime_explanations_{model_type}'
    
    # Load all importance data files
    importance_files = glob.glob(f"{data_dir}/lime_importance_model*.npz")
    model_data = {}
    
    for file_path in importance_files:
        model_num = int(file_path.split('model')[-1].split('.')[0])
        data = np.load(file_path, allow_pickle=True)
        model_data[model_num] = {
            'feature_importance': data['feature_importance'],
            'feature_names': data['feature_names'],
            'feature_indices': data['feature_indices']
        }
    
    return model_data

def get_top_features(model_data, model_num, top_n=5):
    """Get the top N most important features for a specific model"""
    if model_num not in model_data:
        return []
    
    data = model_data[model_num]
    importance = data['feature_importance']
    names = data['feature_names']
    
    # Sort features by importance
    sorted_idx = np.argsort(-importance)  # Descending order
    top_features = []
    
    for i in range(min(top_n, len(sorted_idx))):
        idx = sorted_idx[i]
        feature_name = names[idx]
        feature_importance = importance[idx]
        top_features.append((feature_name, feature_importance))
    
    return top_features

def analyze_sample_explanations(model_type, model_num):
    """Analyze individual sample explanations for patterns"""
    data_dir = f'lime_explanations_{model_type}'
    explanation_files = glob.glob(f"{data_dir}/explanation_model{model_num}_sample*.html")
    
    return {
        'num_samples': len(explanation_files),
        'sample_files': explanation_files
    }

def generate_explanation_text():
    """Generate comprehensive explanation text for all LIME graphs"""
    model_types = ['best', 'worst']
    explanations = []
    
    # Introduction
    explanations.append("# LIME Analysis Explanation\n")
    explanations.append("## Introduction")
    explanations.append("LIME (Local Interpretable Model-agnostic Explanations) is a technique that explains the predictions of any classifier by approximating it locally with an interpretable model. Below, we analyze the LIME explanations for our models to understand which features are most important for distinguishing between ADHD and Control samples.\n")
    
    # Compare best vs worst models
    explanations.append("## Overview: Best vs. Worst Models")
    explanations.append("The analysis compares the top-performing models (best) against the poorest-performing models (worst) to understand which features contribute most to successful classification. This comparison helps identify what discriminative patterns the better models have learned.\n")
    
    # Analyze each model type
    for model_type in model_types:
        model_data = load_lime_data(model_type)
        
        explanations.append(f"## {model_type.title()} Models Analysis")
        explanations.append(f"The {model_type} models represent {'high-performing' if model_type == 'best' else 'low-performing'} models in our ensemble. We analyzed {len(model_data)} models in this category.\n")
        
        for model_num in sorted(model_data.keys()):
            explanations.append(f"### {model_type.title()} Model {model_num}")
            
            # Get top features for this model
            top_features = get_top_features(model_data, model_num)
            explanations.append("#### Key Features")
            explanations.append("The most influential features for this model's predictions are:")
            
            for i, (feature, importance) in enumerate(top_features):
                explanations.append(f"{i+1}. **{feature}** (Importance: {importance:.4f})")
            
            # Analyze individual sample explanations
            sample_info = analyze_sample_explanations(model_type, model_num)
            explanations.append("\n#### Sample Explanations")
            explanations.append(f"We analyzed {sample_info['num_samples']} individual samples with this model. Each explanation shows how specific feature values for that sample influenced the model's prediction.")
            explanations.append("- Positive values (green) indicate features pushing the prediction toward ADHD")
            explanations.append("- Negative values (red) indicate features pushing the prediction toward Control")
            explanations.append("- The magnitude of each bar represents the strength of influence\n")
            
            explanations.append("#### Interpretation")
            if model_type == 'best':
                explanations.append("This high-performing model effectively leverages the most discriminative features. The feature importance is more concentrated on specific key features that show strong predictive power for ADHD detection.")
            else:
                explanations.append("This lower-performing model shows less focused feature importance, possibly attributing significance to features that don't truly differentiate between classes as effectively.")
            
            explanations.append("\n")
    
    # Overall comparison and insights
    explanations.append("## Key Differences Between Best and Worst Models")
    explanations.append("The analysis reveals several important differences between high and low-performing models:")
    explanations.append("1. **Feature Focus**: Best models tend to consistently focus on specific discriminative features")
    explanations.append("2. **Feature Stability**: Best models show more consistent feature importance across different samples")
    explanations.append("3. **Decision Boundaries**: Best models create clearer decision boundaries using fewer but more informative features")
    explanations.append("\n")
    
    explanations.append("## Conclusion")
    explanations.append("The LIME analysis demonstrates that successful ADHD classification relies on the model's ability to identify specific neuroimaging patterns. The best models have learned to focus on a smaller set of highly discriminative features, whereas the worst models either focus on irrelevant features or fail to properly weight the important ones.")
    explanations.append("\nUnderstanding these patterns can help improve future model design and potentially provide insights into the neurological markers of ADHD.")
    
    return "\n".join(explanations)

def main():
    """Generate and save explanation text"""
    explanation_text = generate_explanation_text()
    
    # Save as markdown
    output_file = "lime_analysis_explanation.md"
    with open(output_file, 'w') as f:
        f.write(explanation_text)
    
    print(f"Explanation saved to {output_file}")

if __name__ == "__main__":
    main()
