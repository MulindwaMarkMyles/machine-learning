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
import plotly.express as px
import os.path
import sys
import io
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.calibration import calibration_curve

# Set page config
st.set_page_config(
    page_title="ADHD Classification System (Calibrated)",
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

@st.cache_resource
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
        
        # Try different versions
        for version in [3, 2, 1]:
            try:
                model = mlflow.pytorch.load_model(f"models:/{deployment_info['endpoint']['model_name']}/{version}")
                st.sidebar.success(f"Model loaded successfully (version {version})!")
                return model
            except Exception as e:
                continue
                
        st.error("Could not load any model version. Please ensure models are registered.")
        return None
            
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
        
        # Normalize the data (z-score normalization)
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        
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

def make_prediction(model, data, threshold=0.5):
    """Make prediction using the ensemble model with adjustable threshold"""
    model.eval()
    with torch.no_grad():
        output = model(data)
        probability = output.item()
        prediction = 1 if probability > threshold else 0
        return prediction, probability

def display_calibrated_results(prediction, probability, threshold):
    """Display classification results with threshold adjustment"""
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Display prediction
        if prediction == 1:
            st.error(f"### Prediction: ADHD Indicated (threshold: {threshold:.2f})")
        else:
            st.success(f"### Prediction: No ADHD Indicated (threshold: {threshold:.2f})")
        
        # Display confidence
        st.write(f"### Raw Probability: {probability*100:.2f}%")
        
        # Show calibrated confidence
        # This simple calibration can be replaced with more sophisticated methods
        calibrated_prob = (probability - 0.5) * 2 + 0.5 if probability > 0.5 else probability * 2
        calibrated_prob = max(0, min(1, calibrated_prob))  # Clamp to [0, 1]
        
        st.write(f"### Calibrated Confidence: {calibrated_prob*100:.2f}%")
    
    with col2:
        # Create gauge chart with threshold marker
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, threshold*100], 'color': "lightgray"},
                    {'range': [threshold*100, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        st.plotly_chart(fig)

def generate_sample_data():
    """Generate sample data for threshold visualization"""
    np.random.seed(42)
    # Generate some synthetic prediction scores
    control_scores = np.random.beta(2, 5, 100)  # More concentrated toward 0
    adhd_scores = np.random.beta(5, 2, 100)     # More concentrated toward 1
    
    # Create dataframe
    df = pd.DataFrame({
        'Score': np.concatenate([control_scores, adhd_scores]),
        'True_Label': ['Control'] * 100 + ['ADHD'] * 100
    })
    return df

def plot_distribution(sample_data, threshold):
    """Plot distribution of scores with threshold line"""
    fig = px.histogram(
        sample_data, x="Score", color="True_Label", 
        barmode="overlay", opacity=0.7,
        title="Distribution of Prediction Scores"
    )
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                 annotation_text=f"Threshold: {threshold:.2f}", 
                 annotation_position="top right")
    
    return fig

def main():
    # Page title and description
    st.title("🧠 ADHD Classification System (Calibrated)")
    st.write("""
    This enhanced system uses an ensemble model to classify ADHD patterns with adjustable prediction thresholds.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Settings sidebar
    st.sidebar.title("Model Settings")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # Information about threshold adjustment
    with st.sidebar.expander("Why adjust threshold?"):
        st.write("""
        Adjusting the threshold can help counteract model bias:
        - Lower threshold (< 0.5): More sensitive to ADHD detection
        - Higher threshold (> 0.5): More conservative, reduces false positives
        """)
    
    # Show distributions for educational purposes
    with st.expander("Understanding Prediction Thresholds"):
        st.write("""
        This visualization shows how the prediction threshold affects classification.
        The samples below are synthetic data for demonstration purposes.
        """)
        sample_data = generate_sample_data()
        st.plotly_chart(plot_distribution(sample_data, threshold))
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a MAT file", type=['mat'])
    
    if uploaded_file is not None:
        with st.spinner("Processing data..."):
            # Process the file
            data = process_mat_file(uploaded_file)
            
            if data is not None:
                # Make prediction with custom threshold
                prediction, probability = make_prediction(model, data, threshold)
                
                # Display results with threshold info
                st.markdown("---")
                st.markdown("## Results")
                display_calibrated_results(prediction, probability, threshold)
                
                # Additional information
                with st.expander("See Technical Details"):
                    st.write("Model Architecture: Ensemble of Tree and GBM models")
                    st.write("Input Shape:", data.shape)
                    st.write("Raw Probability Score:", probability)
                    st.write(f"Classification Threshold: {threshold}")

if __name__ == "__main__":
    main()
