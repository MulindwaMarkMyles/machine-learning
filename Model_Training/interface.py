import torch
import numpy as np
import streamlit as st
import joblib
from torch import nn
import pandas as pd
from scipy.io import loadmat
import os

class EEGNet(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(EEGNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler with caching"""
    try:
        # Load model configuration
        model_config = joblib.load("./models/model_config.pkl")
        input_dim = model_config['input_dim']
        embed_dim = model_config['embed_dim']
        max_rows = model_config['max_rows']
        num_features = model_config['num_features']
        
        model = EEGNet(input_dim=input_dim, embed_dim=embed_dim)
        
        # Try loading best model first, fall back to final model
        try:
            model.load_state_dict(torch.load("./models/best_model_three.pth", map_location="cpu"))
            model_type = "best"
        except:
            model.load_state_dict(torch.load("./models/final_model_three.pth", map_location="cpu"))
            model_type = "final"
            
        scaler = joblib.load("./models/model_three_scaler.pkl")
        model.eval()
        return model, scaler, model_type, max_rows, num_features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

def process_mat_file(file, max_rows, num_features):
    """Process uploaded .mat file"""
    try:
        data_dict = loadmat(file)
        
        # Find the key with the largest data
        max_key = None
        max_size = 0
        for key in data_dict:
            if not key.startswith('__'):
                data = data_dict[key]
                if isinstance(data, np.ndarray) and data.size > max_size:
                    max_key = key
                    max_size = data.size
                    
        if max_key is None:
            raise ValueError("No valid data found in .mat file")
            
        data = data_dict[max_key]
        
        # Ensure consistent dimensions with training data
        if data.shape[1] != num_features:
            raise ValueError(f"Expected {num_features} features, but got {data.shape[1]}")
            
        # Pad or truncate to match training dimensions
        if data.shape[0] < max_rows:
            padding = np.zeros((max_rows - data.shape[0], data.shape[1]))
            data = np.vstack((data, padding))
        elif data.shape[0] > max_rows:
            data = data[:max_rows, :]
            
        return data
    except Exception as e:
        raise ValueError(f"Error processing .mat file: {str(e)}")

def main():
    st.set_page_config(page_title="EEG ADHD Classification", page_icon="🧠", layout="wide")
    
    st.title("🧠 EEG ADHD Classification Interface")
    st.markdown("---")
    
    # Load model and scaler
    model, scaler, model_type, max_rows, num_features = load_model_and_scaler()
    if model is None:
        st.error("Failed to load model. Please check model files.")
        return
        
    st.info(f"Using {model_type} model for predictions")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        This interface accepts:
        - Single .mat files
        - Batch .mat files
        - CSV files with 19 columns
        
        Expected data format:
        - 19 features per timepoint
        - Will automatically handle padding
        """)
        
    # File upload section
    st.header("📁 Data Input")
    upload_type = st.radio("Select input type:", ["Single .mat file", "Batch .mat files", "CSV file"])
    
    if upload_type == "Single .mat file":
        uploaded_file = st.file_uploader("Upload .mat file", type=["mat"])
        if uploaded_file:
            try:
                data = process_mat_file(uploaded_file, max_rows, num_features)
                with st.spinner("Processing..."):
                    # Preprocess and predict
                    data = scaler.transform(data)
                    data_tensor = torch.FloatTensor(data.flatten()).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(data_tensor)
                        confidence = output.item()
                        predicted_class = 1 if confidence > 0.5 else 0
                
                # Show results with visual feedback
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", "ADHD" if predicted_class == 1 else "Control")
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Visualization
                st.progress(confidence)
            except:
                st.error("Error processing .mat file")
                
    elif upload_type == "Batch .mat files":
        uploaded_files = st.file_uploader("Upload .mat files", type=["mat"], accept_multiple_files=True)
        if uploaded_files:
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                try:
                    data = process_mat_file(file, max_rows, num_features)
                    data = scaler.transform(data)
                    data_tensor = torch.FloatTensor(data.flatten()).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(data_tensor)
                        confidence = output.item()
                        predicted_class = 1 if confidence > 0.5 else 0
                        
                    results.append({
                        "File": file.name,
                        "Prediction": "ADHD" if predicted_class == 1 else "Control",
                        "Confidence": f"{confidence:.2%}"
                    })
                    
                except Exception as e:
                    results.append({
                        "File": file.name,
                        "Prediction": "Error",
                        "Confidence": str(e)
                    })
                    
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            st.dataframe(pd.DataFrame(results))
            
    else:  # CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if df.shape[1] != num_features:
                    st.error(f"CSV must have exactly {num_features} columns")
                    return
                    
                results = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    # Reshape and pad data to match training dimensions
                    data = row.values.reshape(1, -1)  # First reshape to 2D
                    # Pad to match training dimensions
                    padded_data = np.zeros((max_rows, num_features))
                    padded_data[0] = data[0]  # Put the single row at the start
                    
                    # Apply scaling
                    scaled_data = scaler.transform(padded_data)
                    # Flatten to match model input dimension
                    data_tensor = torch.FloatTensor(scaled_data.flatten()).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(data_tensor)
                        confidence = output.item()
                        predicted_class = 1 if confidence > 0.5 else 0
                        
                    results.append({
                        "Row": i + 1,
                        "Prediction": "ADHD" if predicted_class == 1 else "Control",
                        "Confidence": f"{confidence:.2%}"
                    })
                    
                    progress_bar.progress((i + 1) / len(df))
                    
                st.dataframe(pd.DataFrame(results))
                
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                st.error(f"Shape info - Expected: {max_rows}x{num_features}, Got: {df.shape}")

if __name__ == "__main__":
    main()