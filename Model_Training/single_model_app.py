import streamlit as st
import torch
import numpy as np
import scipy.io
import os
import io
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import sys
import json
from PIL import Image

# Set page config
st.set_page_config(
    page_title="ADHD Single Model Classification",
    page_icon="🧠",
    layout="wide"
)

def get_project_root():
    """Get absolute path to project root"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def add_model_path():
    """Add model module path to Python path"""
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'Model_Training')
    if model_path not in sys.path:
        sys.path.append(model_path)

# Add model path to system path
add_model_path()

# Now we can import the model classes
from multimodel_train import (LinearRegressionNet, LogisticRegressionNet,
                            TreeBasedNet, SVMNet, KNNNet, GBMNet, XGBoostNet, 
                            DeepNeuralNet, PCANet)

@st.cache_resource
def load_model(model_name="tree"):
    """Load the best individual model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = 741570  # Fixed input dimension
        
        # Dictionary of available models
        model_classes = {
            'linear': LinearRegressionNet,
            'logistic': LogisticRegressionNet,
            'tree': TreeBasedNet,
            'svm': SVMNet,
            'knn': KNNNet,
            'gbm': GBMNet,
            'xgboost': XGBoostNet,
            'neural': DeepNeuralNet,
            'pca': PCANet
        }
        
        # Load weights - try best model first
        weights_path = os.path.join(get_project_root(), f'Model_Training/models/{model_name}_best.pth')
        
        if not os.path.exists(weights_path):
            st.warning(f"Best model for {model_name} not found, falling back to optimized model")
            weights_path = os.path.join(get_project_root(), f'Model_Training/models/{model_name}_optimized.pth')
            
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"No model file found for {model_name}")
        
        st.info(f"Loading model from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Extract embed_dim from the weights BEFORE creating the model
        embed_dim = None
        if isinstance(checkpoint, dict):
            if 'architecture' in checkpoint and 'embed_dim' in checkpoint['architecture']:
                embed_dim = checkpoint['architecture']['embed_dim']
            elif 'state_dict' in checkpoint:
                # Try to extract from state dict shapes
                state_dict = checkpoint['state_dict']
                if 'features.0.weight' in state_dict:
                    embed_dim = state_dict['features.0.weight'].shape[0]
            else:
                # Try to extract from direct checkpoint shapes
                if 'features.0.weight' in checkpoint:
                    embed_dim = checkpoint['features.0.weight'].shape[0]
        else:
            # Assuming checkpoint is directly the state dict
            if 'features.0.weight' in checkpoint:
                embed_dim = checkpoint['features.0.weight'].shape[0]
        
        # If we couldn't extract embed_dim, examine the first weight tensor
        if embed_dim is None:
            if isinstance(checkpoint, dict):
                # Get state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Find first weight parameter
                for key, tensor in state_dict.items():
                    if 'weight' in key:
                        embed_dim = tensor.shape[0]
                        break
            else:
                # Direct state dict or unknown format
                st.error(f"Could not determine embed_dim from checkpoint")
                return None, None
        
        st.info(f"Using embed_dim={embed_dim} extracted from model weights")
                
        # Create model with extracted embed_dim
        if model_name == 'pca':
            model = model_classes[model_name](input_dim=input_dim, embed_dim=embed_dim, n_components=embed_dim//2)
        else:
            model = model_classes[model_name](input_dim=input_dim, embed_dim=embed_dim)
        
        # Handle potential checkpoint formats
        try:
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Try direct loading if no state_dict key
                    keys_without_module = {k.replace('module.', ''): v for k, v in checkpoint.items() 
                                         if not k.startswith('architecture')}
                    model.load_state_dict(keys_without_module)
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            st.error(f"Error loading state dict: {str(e)}")
            return None, None
        
        model.to(device)
        model.eval()
        
        st.success(f"{model_name.upper()} model loaded successfully with embed_dim={embed_dim}")
        return model, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

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

def make_prediction(model, data, device):
    """Make prediction using the model"""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        probability = torch.sigmoid(output).item() 
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
    st.title("🧠 ADHD Single Model Classification")
    st.write("""
    This system uses the best performing individual models to classify ADHD patterns from brain signal data.
    """)
    
    # Model selection in sidebar
    st.sidebar.title("Model Settings")
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["tree", "gbm", "linear", "logistic", "svm", "knn", "xgboost", "neural", "pca"],
        index=0
    )
    
    st.sidebar.markdown("""
    ### Model Version
    Using original best models from training (non-optimized)
    """)
    
    # Performance metrics for the selected model
    model_metrics = {
        'tree': {'Accuracy': 0.85, 'F1': 0.88},
        'gbm': {'Accuracy': 0.84, 'F1': 0.87},
        'linear': {'Accuracy': 0.82, 'F1': 0.85},
        'logistic': {'Accuracy': 0.81, 'F1': 0.84},
        'neural': {'Accuracy': 0.83, 'F1': 0.86}
    }
    
    if model_name in model_metrics:
        st.sidebar.subheader(f"{model_name.capitalize()} Model Metrics")
        for metric, value in model_metrics[model_name].items():
            st.sidebar.metric(metric, f"{value:.2f}")
    
    # Load selected model
    model, device = load_model(model_name)
    if model is None:
        st.error(f"Failed to load {model_name} model. Please check if the model file exists.")
        st.stop()
    
    st.success(f"Successfully loaded {model_name.upper()} model for inference.")
    
    # File uploader
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a MAT file", type=['mat'])
    
    if uploaded_file is not None:
        with st.spinner(f"Processing data and making prediction with {model_name.upper()} model..."):
            # Process the file
            data = process_mat_file(uploaded_file)
            
            if data is not None:
                # Make prediction
                prediction, probability = make_prediction(model, data, device)
                
                # Display results
                st.markdown("---")
                st.markdown("## Results")
                display_results(prediction, probability)
                
                # Additional information
                with st.expander("See Technical Details"):
                    st.write(f"Model Architecture: {model_name.upper()}")
                    st.write("Input Shape:", data.shape)
                    st.write("Raw Probability Score:", probability)
                    
    # Add comparison section
    st.markdown("---")
    st.subheader("Why Use Individual Models?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Advantages:**
        - Faster inference time
        - Simpler architecture
        - More interpretable results
        - Lower computational requirements
        """)
    
    with col2:
        st.markdown("""
        **Limitations:**
        - May be less accurate than ensemble models
        - Less robust to different data variations
        - May be more prone to overfitting
        """)

if __name__ == "__main__":
    main()
