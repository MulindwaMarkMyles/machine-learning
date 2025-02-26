import streamlit as st
import mlflow
import mlflow.pytorch
import torch
import numpy as np
import scipy.io
import json
import os
from PIL import Image
import plotly.graph_objects as go
import os.path
import sys
import io
from scipy.interpolate import interp1d

# Set page config
st.set_page_config(
    page_title="ADHD Classification System",
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

def load_deployment_info():
    """Load deployment information from JSON"""
    project_root = get_project_root()
    info_path = os.path.join(project_root, 'Model_Training/models/ensemble_deployment_info.json')
    if not os.path.exists(info_path):
        st.error("Deployment info not found!")
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)

def load_model():
    """Load the ensemble model from MLflow"""
    try:
        # Add model module path
        add_model_module_path()
        
        deployment_info = load_deployment_info()
        if deployment_info is None:
            return None
        
        # Setup MLflow with absolute path
        project_root = get_project_root()
        mlflow_db = os.path.join(project_root, 'Model_Training/mlflow.db')
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        
        # Load model version 2
        try:
            model = mlflow.pytorch.load_model(f"models:/{deployment_info['endpoint']['model_name']}/3")
            st.sidebar.success("Model loaded successfully!")
            
            # Display model performance metrics
            st.sidebar.markdown("### Model Performance Metrics")
            metrics = {
                'Accuracy': 0.8947,
                'Precision': 0.8571,
                'Recall': 1.0000,
                'F1 Score': 0.9231,
                'Specificity': 0.7143
            }
            
            for metric, value in metrics.items():
                st.sidebar.metric(metric, f"{value:.4f}")
            
            st.sidebar.markdown("### Model Architecture")
            st.sidebar.write("- Ensemble of Tree and GBM models")
            st.sidebar.write("- Base models: ['tree', 'gbm']")
            
        except Exception as e:
            st.error("Model not found in MLflow. Please ensure the model is registered.")
            st.error(f"Details: {str(e)}")
            st.info("Try running the deployment script (mlflow_deploy_ensemble.py) first.")
            return None
            
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_mat_file(uploaded_file):
    """Process uploaded MAT file to ensure exact dimensions for model input"""
    try:
        # These must match EXACTLY what the model expects
        target_time_points = 1029
        target_features = 721
        expected_dim = 741570  # Exact value, not calculated
        
        # Read and extract data
        bytes_data = io.BytesIO(uploaded_file.getvalue())
        mat_data = scipy.io.loadmat(bytes_data)
        data = mat_data['data'] if 'data' in mat_data else next(
            value for key, value in mat_data.items() 
            if not key.startswith('__') and isinstance(value, np.ndarray)
        )
        
        st.write("Original data shape:", data.shape)
        
        # Create a correctly sized output array - use the exact dimensions
        final_data = np.zeros((1, expected_dim))
        
        # Interpolate time points for each channel
        original_samples, n_channels = data.shape
        original_time = np.linspace(0, 1, original_samples)
        target_time = np.linspace(0, 1, target_time_points)
        
        # Only process as many channels as target_features (721)
        n_channels_to_process = min(n_channels, target_features)
        
        # Fill the final data array channel by channel
        for i in range(n_channels_to_process):
            # Interpolate this channel's time points
            interpolator = interp1d(original_time, data[:, i], kind='linear')
            interpolated_channel = interpolator(target_time)
            
            # Calculate exact position in the final flattened array
            for t in range(target_time_points):
                final_data[0, i + (t * target_features)] = interpolated_channel[t]
        
        # Convert to tensor and verify shape
        tensor_data = torch.FloatTensor(final_data)
        
        # Final verification
        if tensor_data.shape[1] != expected_dim:
            st.error(f"CRITICAL ERROR: Final shape {tensor_data.shape[1]} != expected {expected_dim}")
            return None
            
        st.success(f"Data processed successfully. Shape: {tensor_data.shape}")
        
        return tensor_data
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Data structure found in file:")
        if 'mat_data' in locals():
            st.write({k: v.shape if isinstance(v, np.ndarray) else type(v) 
                     for k, v in mat_data.items() if not k.startswith('__')})
        return None

def make_prediction(model, data):
    """Make prediction using the ensemble model"""
    model.eval()
    with torch.no_grad():
        output = model(data)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
        return prediction, probability

def display_results(prediction, probability):
    """Display classification results with visualizations"""
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Display prediction
        if prediction == 1:
            st.error("### Prediction: ADHD Indicated")
        else:
            st.success("### Prediction: No ADHD Indicated")
        
        # Display confidence
        st.write(f"### Confidence: {probability*100:.2f}%")
    
    with col2:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig)

def main():
    # Page title and description
    st.title("🧠 ADHD Classification System")
    st.write("""
    This system uses an ensemble of Tree and GBM models to classify ADHD patterns from brain signal data.
    The model achieves 89.47% accuracy and 92.31% F1 score on validation data.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a MAT file", type=['mat'])
    
    if uploaded_file is not None:
        with st.spinner("Processing data..."):
            # Process the file
            data = process_mat_file(uploaded_file)
            
            if data is not None:
                # Make prediction
                prediction, probability = make_prediction(model, data)
                
                # Display results
                st.markdown("---")
                st.markdown("## Results")
                display_results(prediction, probability)
                
                # Additional information
                with st.expander("See Technical Details"):
                    st.write("Model Architecture: Ensemble of Tree and GBM models")
                    st.write("Input Shape:", data.shape)
                    st.write("Raw Probability Score:", probability)

if __name__ == "__main__":
    main()
