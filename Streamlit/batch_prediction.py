import streamlit as st
import mlflow
import mlflow.pytorch
import torch
import numpy as np
import scipy.io
import json
import os
import io
import sys
from scipy.interpolate import interp1d
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set page config
st.set_page_config(
    page_title="ADHD Batch Prediction",
    page_icon="🧠",
    layout="wide"
)

def get_project_root():
    """Get absolute path to project root"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def add_model_module_path():
    """Add model module path to Python path"""
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'Model_Training')
    if model_path not in sys.path:
        sys.path.append(model_path)

@st.cache_resource
def load_model():
    """Load the model from MLflow"""
    try:
        # Add model module path
        add_model_module_path()
        
        # Get deployment info
        project_root = get_project_root()
        info_path = os.path.join(project_root, 'Model_Training/models/ensemble_deployment_info.json')
        if not os.path.exists(info_path):
            st.error("Deployment info not found!")
            return None
        
        with open(info_path, 'r') as f:
            deployment_info = json.load(f)
        
        # Setup MLflow
        mlflow_db = os.path.join(project_root, 'Model_Training/mlflow.db')
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        
        # Try different versions
        for version in [3, 2, 1]:
            try:
                model = mlflow.pytorch.load_model(f"models:/{deployment_info['endpoint']['model_name']}/{version}")
                st.sidebar.success(f"Model loaded successfully (version {version})!")
                return model
            except Exception:
                continue
                
        st.error("Could not load any model version")
        return None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_mat_file(file_content):
    """Process MAT file content to ensure exact dimensions for model input"""
    try:
        # These must match EXACTLY what the model expects
        target_time_points = 1029
        target_features = 721
        expected_dim = 741570  # Exact value, not calculated
        
        # Read and extract data
        bytes_data = io.BytesIO(file_content)
        mat_data = scipy.io.loadmat(bytes_data)
        data = mat_data['data'] if 'data' in mat_data else next(
            value for key, value in mat_data.items() 
            if not key.startswith('__') and isinstance(value, np.ndarray)
        )
        
        # Normalize the data
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        
        # Create a correctly sized output array
        final_data = np.zeros((1, expected_dim))
        
        # Interpolate time points for each channel
        original_samples, n_channels = data.shape
        original_time = np.linspace(0, 1, original_samples)
        target_time = np.linspace(0, 1, target_time_points)
        
        # Only process as many channels as target_features
        n_channels_to_process = min(n_channels, target_features)
        
        # Fill the final data array channel by channel
        for i in range(n_channels_to_process):
            interpolator = interp1d(original_time, data[:, i], kind='linear')
            interpolated_channel = interpolator(target_time)
            
            for t in range(target_time_points):
                final_data[0, i + (t * target_features)] = interpolated_channel[t]
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(final_data)
        
        # Final verification
        if tensor_data.shape[1] != expected_dim:
            return None
            
        return tensor_data
    
    except Exception as e:
        return None

def make_prediction(model, data):
    """Make prediction using the model"""
    model.eval()
    with torch.no_grad():
        output = model(data)
        probability = output.item()
        return probability

def analyze_probabilities(probabilities):
    """Analyze probability distribution for bias"""
    mean_prob = np.mean(probabilities)
    median_prob = np.median(probabilities)
    
    analysis = {
        'mean': mean_prob,
        'median': median_prob,
        'min': min(probabilities),
        'max': max(probabilities),
        'adhd_percent': sum(p > 0.5 for p in probabilities) / len(probabilities) * 100,
        'control_percent': sum(p <= 0.5 for p in probabilities) / len(probabilities) * 100
    }
    
    # Determine if there might be bias
    if mean_prob > 0.7:
        analysis['bias'] = "Model may be biased toward ADHD predictions"
    elif mean_prob < 0.3:
        analysis['bias'] = "Model may be biased toward control predictions"
    else:
        analysis['bias'] = "No strong bias detected"
        
    return analysis

def find_optimal_threshold(true_labels, probabilities):
    """Find optimal threshold using ROC curve"""
    if len(set(true_labels)) < 2:
        return 0.5, None  # Need both classes
    
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold that maximizes J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    # Create ROC curve figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[fpr[best_idx]], y=[tpr[best_idx]], mode='markers', 
                           marker=dict(size=10, color='red'), name=f'Optimal ({optimal_threshold:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
    
    fig.update_layout(
        title='ROC Curve with Optimal Threshold',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate')
    )
    
    return optimal_threshold, fig

def main():
    st.title("🧠 ADHD Batch Prediction & Bias Analysis")
    st.write("""
    Upload multiple files to:
    1. Check model bias by analyzing prediction distributions
    2. Find optimal threshold for classification
    3. Generate model performance metrics
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar settings
    st.sidebar.title("Analysis Options")
    has_labels = st.sidebar.checkbox("Data has known labels", value=False)
    
    if has_labels:
        label_map = st.sidebar.radio(
            "How are files labeled?",
            ["Filename contains class", "Provide CSV with labels"]
        )
        
        if label_map == "Filename contains class":
            adhd_keyword = st.sidebar.text_input("ADHD keyword in filename", "adhd")
            control_keyword = st.sidebar.text_input("Control keyword in filename", "control")
    
    # File upload section
    uploaded_files = st.file_uploader("Upload multiple .mat files", type=['mat'], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} files...")
        progress = st.progress(0)
        
        results = []
        probabilities = []
        true_labels = []
        
        # Process each file
        for i, file in enumerate(uploaded_files):
            try:
                # Process file
                data = process_mat_file(file.getvalue())
                
                if data is not None:
                    # Make prediction
                    probability = make_prediction(model, data)
                    
                    # Get true label if available
                    true_label = None
                    if has_labels and label_map == "Filename contains class":
                        if adhd_keyword.lower() in file.name.lower():
                            true_label = 1
                        elif control_keyword.lower() in file.name.lower():
                            true_label = 0
                    
                    # Store results
                    results.append({
                        'filename': file.name,
                        'probability': probability,
                        'prediction': 1 if probability > 0.5 else 0,
                        'true_label': true_label
                    })
                    
                    probabilities.append(probability)
                    if true_label is not None:
                        true_labels.append(true_label)
                
                else:
                    results.append({
                        'filename': file.name,
                        'probability': None,
                        'prediction': None,
                        'true_label': None,
                        'error': 'Processing failed'
                    })
            
            except Exception as e:
                results.append({
                    'filename': file.name,
                    'probability': None, 
                    'prediction': None,
                    'true_label': None,
                    'error': str(e)
                })
            
            # Update progress
            progress.progress((i + 1) / len(uploaded_files))
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(results)
        
        # Display results
        st.markdown("## Results")
        
        if len(probabilities) > 0:
            # Analyze probabilities
            analysis = analyze_probabilities(probabilities)
            
            # Display analysis
            st.markdown("### Probability Distribution Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Probability", f"{analysis['mean']:.3f}")
            with col2:
                st.metric("Median Probability", f"{analysis['median']:.3f}")
            with col3:
                st.metric("ADHD/Control Ratio", f"{analysis['adhd_percent']:.1f}% / {analysis['control_percent']:.1f}%")
            
            st.info(analysis['bias'])
            
            # Plot probability distribution
            fig = px.histogram(
                df, x="probability", 
                title="Distribution of Prediction Probabilities",
                labels={"probability": "Probability", "count": "Count"}
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig)
            
            # Calculate optimal threshold if labels are available
            if len(true_labels) > 0:
                st.markdown("### Threshold Optimization")
                
                optimal_threshold, roc_fig = find_optimal_threshold(true_labels, 
                                                   [p for p, l in zip(probabilities, true_labels) if l is not None])
                
                if roc_fig is not None:
                    st.plotly_chart(roc_fig)
                    
                    st.success(f"Optimal threshold: {optimal_threshold:.3f}")
                    
                    # Recalculate predictions with optimal threshold
                    optimal_preds = [1 if p > optimal_threshold else 0 for p in probabilities]
                    
                    # Create confusion matrix
                    cm = confusion_matrix(true_labels, optimal_preds)
                    
                    # Display confusion matrix
                    st.markdown("### Confusion Matrix (with optimal threshold)")
                    fig = px.imshow(
                        cm, 
                        text_auto=True,
                        labels=dict(x="Predicted", y="True"),
                        x=['Control', 'ADHD'],
                        y=['Control', 'ADHD']
                    )
                    st.plotly_chart(fig)
                    
                    # Calculate metrics
                    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
                    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
                    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.3f}")
                    col2.metric("Precision", f"{precision:.3f}")
                    col3.metric("Recall", f"{recall:.3f}")
                    col4.metric("F1 Score", f"{f1:.3f}")
                    
            # Display detailed results table
            st.markdown("### Detailed Results")
            st.dataframe(df)
            
            # Allow download of results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results as CSV",
                csv,
                "prediction_results.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.error("No valid predictions could be made")

if __name__ == "__main__":
    main()
