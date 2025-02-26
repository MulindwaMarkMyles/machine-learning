import streamlit as st
import torch
import numpy as np
import scipy.io
import os
import io
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import plotly.express as px
import sys
import json
import joblib
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Set page config
st.set_page_config(
    page_title="ADHD Classification (Balanced Models)",
    page_icon="🧠",
    layout="wide"
)

# Helper functions
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

# Import needed modules
sys.path.append(os.path.join(get_project_root(), 'Model_Training'))
from multimodel_train import (LinearRegressionNet, TreeBasedNet, GBMNet)
from balanced_retraining import EnsembleModel

# Load normalization parameters
@st.cache_resource
def load_normalization():
    try:
        norm_params = joblib.load(os.path.join(get_project_root(), 
                                              'Model_Training/models/normalization_params.pkl'))
        # Check format of the normalization parameters
        if isinstance(norm_params, dict):
            # Handle different dictionary structures
            if 'scaler' in norm_params:
                return {'is_scaler': True, 'scaler': norm_params['scaler']}
            elif 'mean' in norm_params and 'std' in norm_params:
                return {'is_scaler': False, 'mean': norm_params['mean'], 'std': norm_params['std']}
            else:
                st.warning("Unknown normalization parameter format. Using default normalization.")
                return None
        else:
            st.warning("Normalization parameters not in expected format. Using default normalization.")
            return None
    except Exception as e:
        st.warning(f"Error loading normalization parameters: {str(e)}. Using default normalization.")
        return None

@st.cache_resource
def load_model(model_name="linear_tree_ensemble"):
    """Load the balanced model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Path to model file
        model_path = os.path.join(get_project_root(), 
                                  f'Model_Training/models/{model_name}_balanced.pth')
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, device, None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model architecture
        if 'architecture' in checkpoint:
            input_dim = checkpoint['architecture']['input_dim']
            embed_dim = checkpoint['architecture']['embed_dim']
            
            # Store the required input dimension for validation later
            required_input_dim = input_dim
            
            if 'models' in checkpoint['architecture']:
                # This is an ensemble model
                models = []
                model_names = checkpoint['architecture']['models']
                
                for name in model_names:
                    if name == 'linear':
                        models.append(LinearRegressionNet(input_dim=input_dim, embed_dim=embed_dim))
                    elif name == 'tree':
                        models.append(TreeBasedNet(input_dim=input_dim, embed_dim=embed_dim))
                    elif name == 'gbm':
                        models.append(GBMNet(input_dim=input_dim, embed_dim=embed_dim))
                
                model = EnsembleModel(models, device)
            else:
                # Single model
                if 'linear' in model_name:
                    model = LinearRegressionNet(input_dim=input_dim, embed_dim=embed_dim)
                elif 'tree' in model_name:
                    model = TreeBasedNet(input_dim=input_dim, embed_dim=embed_dim)
                elif 'gbm' in model_name:
                    model = GBMNet(input_dim=input_dim, embed_dim=embed_dim)
                else:
                    st.error(f"Unknown model type: {model_name}")
                    return None, device, None
        else:
            st.error("Model architecture information missing")
            return None, device, None
                
        # Load state dictionary
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        return model, device, required_input_dim
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device, None

def process_mat_file(uploaded_file, norm_params=None, required_input_dim=None):
    """Process MAT file with proper normalization"""
    try:
        # Read data
        bytes_data = io.BytesIO(uploaded_file.getvalue())
        mat_data = scipy.io.loadmat(bytes_data)
        
        # Extract data array
        if 'data' in mat_data:
            data = mat_data['data']
        else:
            # Find any array with appropriate dimensions
            arrays_found = []
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if len(value.shape) == 2:  # Must be 2D
                        arrays_found.append((key, value))
            
            if not arrays_found:
                raise ValueError("No suitable 2D array found in the MAT file")
            
            # Use the largest array (by total elements)
            key, data = max(arrays_found, key=lambda x: x[1].size)
            st.info(f"Using array with key '{key}' from the MAT file")
        
        st.write("Original data shape:", data.shape)
        
        # Apply normalization if available
        if norm_params is not None:
            if 'is_scaler' in norm_params:
                if norm_params['is_scaler']:
                    # Use scikit-learn scaler
                    scaler = norm_params['scaler']
                    data = scaler.transform(data)
                    st.info("Applied RobustScaler normalization")
                else:
                    # Use mean/std normalization
                    try:
                        mean = norm_params['mean']
                        std = norm_params['std']
                        
                        # Check if dimensions match
                        if len(mean) == data.shape[1]:
                            data = (data - mean) / std
                            st.info("Applied pre-computed mean/std normalization")
                        else:
                            # Fall back to per-channel normalization
                            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                            st.info("Dimensions mismatch - applied on-the-fly normalization instead")
                    except Exception as e:
                        st.warning(f"Error applying normalization: {str(e)}. Using on-the-fly normalization.")
                        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            else:
                # Use per-channel normalization as fallback
                data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                st.info("Using on-the-fly per-channel normalization")
        else:
            # No normalization parameters - use per-channel normalization
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            st.info("Applied on-the-fly per-channel normalization")
        
        # Process data to match model dimensions
        if required_input_dim:
            st.info(f"Reshaping data to match model's required input dimension: {required_input_dim}")
            
            # Calculate how to reshape based on the available data
            original_samples, n_features = data.shape
            total_elements = original_samples * n_features
            
            # Flatten first to get all data elements
            flattened_data = data.flatten()
            
            # Handle the case where we have more or fewer elements than needed
            if total_elements > required_input_dim:
                # Truncate to required dimension
                st.warning(f"Truncating input data from {total_elements} to {required_input_dim} elements")
                tensor_data = torch.FloatTensor(flattened_data[:required_input_dim]).unsqueeze(0)
            elif total_elements < required_input_dim:
                # Pad with zeros to reach required dimension
                st.warning(f"Padding input data from {total_elements} to {required_input_dim} elements")
                padded_data = np.zeros(required_input_dim)
                padded_data[:total_elements] = flattened_data
                tensor_data = torch.FloatTensor(padded_data).unsqueeze(0)
            else:
                # Exact match
                tensor_data = torch.FloatTensor(flattened_data).unsqueeze(0)
        else:
            # Default processing without specific dimension requirement
            # Just flatten the data
            tensor_data = torch.FloatTensor(data.flatten()).unsqueeze(0)
            
        st.success(f"Data processed successfully. Shape: {tensor_data.shape}")
        
        return tensor_data
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def make_prediction(model, data, device, threshold=0.5):
    """Make prediction with threshold"""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        probability = output.item()
        prediction = 1 if probability > threshold else 0
        return prediction, probability

def display_results(prediction, probability, threshold):
    """Display classification results with threshold information"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Display prediction with threshold context
        if prediction == 1:
            st.error(f"### Prediction: ADHD Indicated (threshold: {threshold:.2f})")
        else:
            st.success(f"### Prediction: No ADHD Indicated (threshold: {threshold:.2f})")
        
        # Display confidence
        st.write(f"### Raw Probability: {probability*100:.2f}%")
        
        # Distance from threshold
        threshold_difference = abs(probability - threshold)
        confidence_text = "High" if threshold_difference > 0.2 else \
                         "Medium" if threshold_difference > 0.1 else "Low"
                         
        st.write(f"### Prediction Confidence: {confidence_text}")
        st.write(f"### Distance from threshold: {threshold_difference*100:.2f}%")
    
    with col2:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, threshold*100], 'color': "lightgray"},
                    {'range': [threshold*100, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold*100
                }
            }
        ))
        st.plotly_chart(fig)

def main():
    st.title("🧠 ADHD Classification System (Balanced Models)")
    st.write("""
    This application uses models that were specifically trained to avoid bias toward ADHD predictions.
    These balanced models provide more reliable classification results.
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["linear", "tree", "gbm", "linear_tree_ensemble", "linear_gbm_ensemble"],
        index=3  # Default to linear_tree_ensemble
    )
    
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust this threshold to control the sensitivity of ADHD detection."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About the Models
    These models were trained with special balancing techniques to reduce bias:
    
    - **linear**: Linear neural network
    - **tree**: Tree-based neural network
    - **gbm**: Gradient boosting neural network
    - **linear_tree_ensemble**: Ensemble of linear and tree models
    - **linear_gbm_ensemble**: Ensemble of linear and gradient boosting models
    """)
    
    # Load model based on selection and get required input dimension
    model, device, required_input_dim = load_model(model_name)
    
    # Load normalization parameters
    norm_params = load_normalization()

    # File uploader
    st.subheader("Upload Brain Activity Data")
    
    uploaded_file = st.file_uploader("Choose a MAT file", type=['mat'])
    
    col1, col2 = st.columns(2)
    
    # Add the missing prediction logic
    if uploaded_file is not None:
        # Process the file with required dimension
        with st.spinner("Processing data..."):
            tensor_data = process_mat_file(uploaded_file, norm_params, required_input_dim)
            
        if tensor_data is not None:
            # Verify dimensions before prediction
            if required_input_dim and tensor_data.shape[1] != required_input_dim:
                st.error(f"Dimension mismatch. Model expects {required_input_dim} elements but got {tensor_data.shape[1]}.")
            else:
                # Make prediction
                with st.spinner("Running prediction..."):
                    prediction, probability = make_prediction(model, tensor_data, device, threshold)
                    
                # Display results
                st.subheader("Prediction Results")
                display_results(prediction, probability, threshold)
                
                # Additional visualization or explanation
                st.subheader("Understanding This Prediction")
                st.write("""
                The prediction above is based on a balanced model that was specifically trained 
                to avoid bias toward either ADHD or neurotypical classifications. 
                
                The threshold can be adjusted in the sidebar to make the model more or less sensitive
                to ADHD indicators in the data.
                """)
    else:
        # No file uploaded yet
        st.info("Please upload a .mat file containing brain activity data to get a prediction.")
        
        # Demo information
        with st.expander("Sample Data Information"):
            st.write("""
            ### Expected Data Format
            
            The model expects brain activity data in a MATLAB .mat file with:
            - Time series data (rows = time points)
            - Features/channels (columns)
            - The data should be stored under the key 'data' in the .mat file
            
            If your data has a different structure, the app will attempt to find the
            largest array in the file and use that.
            """)

if __name__ == "__main__":
    main()
