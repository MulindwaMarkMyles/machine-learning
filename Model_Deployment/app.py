import torch
import numpy as np
import uvicorn
import os
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io, sys
import scipy.io
from typing import Optional

# Import model definition classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model_Training.multimodel_train import (LinearRegressionNet, TreeBasedNet, GBMNet)
from Model_Training.balanced_retraining import EnsembleModel

# Initialize FastAPI app
app = FastAPI(
    title="ADHD Classification API",
    description="API for ADHD classification using balanced models",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from your Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model paths
MODEL_DIR = os.environ.get('MODEL_DIR', './models')
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'linear_gbm_ensemble')
MODEL_PATH = os.path.join(MODEL_DIR, f"{DEFAULT_MODEL}_balanced.pth")
NORM_PARAMS_PATH = os.path.join(MODEL_DIR, "normalization_params.pkl")

# Global variables for loaded models
model = None
device = None
scaler = None
required_input_dim = None

# Request body models
class PredictionRequest(BaseModel):
    threshold: float = 0.5

# Load the model at startup
@app.on_event("startup")
async def load_model():
    global model, device, scaler, required_input_dim
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
        # Load model checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Extract architecture parameters
        if 'architecture' not in checkpoint:
            raise ValueError("Model architecture information missing")
            
        input_dim = checkpoint['architecture']['input_dim']
        embed_dim = checkpoint['architecture']['embed_dim']
        required_input_dim = input_dim
        
        # Initialize appropriate model
        if 'models' in checkpoint['architecture']:
            models = []
            model_names = checkpoint['architecture']['models']
            
            for name in model_names:
                if name == 'linear':
                    models.append(LinearRegressionNet(input_dim=input_dim, embed_dim=embed_dim))
                elif name == 'gbm':
                    models.append(GBMNet(input_dim=input_dim, embed_dim=embed_dim))
                elif name == 'tree':
                    models.append(TreeBasedNet(input_dim=input_dim, embed_dim=embed_dim))
            
            model = EnsembleModel(models, device)
        else:
            raise ValueError("Expected ensemble model but found single model")
            
        # Load model state
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        # Load normalization parameters
        if os.path.exists(NORM_PARAMS_PATH):
            norm_params = joblib.load(NORM_PARAMS_PATH)
            if 'scaler' in norm_params:
                scaler = norm_params['scaler']
                print("Loaded RobustScaler for data normalization")
            else:
                print("Normalization parameters found but no scaler present")
        else:
            print("No normalization parameters found, will use per-feature normalization")
        
        print(f"Model loaded successfully: {DEFAULT_MODEL}")
        print(f"Required input dimension: {required_input_dim}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": DEFAULT_MODEL}

# Prediction endpoint for MAT files
@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    if not file.filename.endswith('.mat'):
        raise HTTPException(status_code=400, detail="Only MAT files are supported")
    
    try:
        # Read the file
        contents = await file.read()
        tensor_data = process_mat_data(contents)
        
        # Make prediction
        with torch.no_grad():
            tensor_data = tensor_data.to(device)
            output = model(tensor_data)
            probability = output.item()
            prediction = 1 if probability > threshold else 0
        
        return {
            "prediction": prediction,
            "probability": probability,
            "threshold": threshold,
            "prediction_label": "ADHD" if prediction == 1 else "No ADHD"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to process MAT data
def process_mat_data(file_bytes):
    try:
        # Read the MAT file from bytes
        bytes_io = io.BytesIO(file_bytes)
        mat_data = scipy.io.loadmat(bytes_io)
        
        # Extract data array
        if 'data' in mat_data:
            data = mat_data['data']
        else:
            # Find any suitable array
            arrays_found = []
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if len(value.shape) == 2:  # Must be 2D
                        arrays_found.append((key, value))
            
            if not arrays_found:
                raise ValueError("No suitable 2D array found in the MAT file")
            
            # Use the largest array
            key, data = max(arrays_found, key=lambda x: x[1].size)
            print(f"Using array with key '{key}' from the MAT file")
        
        print(f"Original data shape: {data.shape}")
        
        # Apply normalization
        if scaler is not None:
            data = scaler.transform(data)
        else:
            # Apply per-feature normalization
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        
        # Process to match required dimensions
        if required_input_dim is not None:
            # Flatten and adjust to required dimension
            flattened_data = data.flatten()
            total_elements = len(flattened_data)
            
            if total_elements > required_input_dim:
                # Truncate
                tensor_data = torch.FloatTensor(flattened_data[:required_input_dim]).unsqueeze(0)
            elif total_elements < required_input_dim:
                # Pad with zeros
                padded_data = np.zeros(required_input_dim)
                padded_data[:total_elements] = flattened_data
                tensor_data = torch.FloatTensor(padded_data).unsqueeze(0)
            else:
                # Exact match
                tensor_data = torch.FloatTensor(flattened_data).unsqueeze(0)
        else:
            tensor_data = torch.FloatTensor(data.flatten()).unsqueeze(0)
        
        return tensor_data
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise e

# Run the server when this script is executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
