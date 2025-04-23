import os
import sys
import torch
import numpy as np
import joblib
import io
import scipy.io
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from waitress import serve
from sklearn.decomposition import PCA
import torch.nn as nn

# Add parent directory to path so we can import our model classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EnsembleModel(nn.Module):
    """Balanced ensemble with adjustable weights"""
    def __init__(self, models, device):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Initialize with balanced weights that sum to 1.0
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        self.device = device
        
        # Freeze base models' parameters - we don't want to train them again
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        predictions = torch.zeros(x.size(0), 1).to(self.device)
        
        # Gather predictions from each model - no torch.no_grad() here
        for i, model in enumerate(self.models):
            predictions += self.weights[i] * model(x)
        
        # Apply sigmoid for balanced output range
        return torch.sigmoid(predictions)


class BaseNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(BaseNetwork, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim


class LinearRegressionNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(LinearRegressionNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class GBMNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(GBMNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)


# Initialize Flask app
app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mat'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model
model = None
device = None
scaler = None
required_input_dim = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_name="linear_gbm_ensemble"):
    """Load the balanced model"""
    global model, device, scaler, required_input_dim
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Path to model file
        model_dir = os.path.join('.')
        model_path = os.path.join(model_dir, f"{model_name}_balanced_optimized.pth")
        norm_params_path = os.path.join(model_dir, "normalization_params.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return False
        
        # First load checkpoint to get architecture info
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Loaded checkpoint, checking architecture...")
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Get model architecture from checkpoint
        if 'architecture' in checkpoint:
            input_dim = checkpoint['architecture']['input_dim']
            # Get embed_dim for each model from hyperparameters if available
            if 'hyperparameters' in checkpoint:
                params = checkpoint['hyperparameters']
                linear_embed_dim = params.get('linear_embed_dim', 85)  # Default to 85 if not found
                gbm_embed_dim = params.get('gbm_embed_dim', 85)  # Default to 85 if not found
            else:
                # Use the saved embed_dim or default to 85
                linear_embed_dim = gbm_embed_dim = 85
            
            required_input_dim = input_dim
            print(f"Loading model with input_dim={input_dim}, linear_embed_dim={linear_embed_dim}, gbm_embed_dim={gbm_embed_dim}")
            
            if 'models' in checkpoint['architecture']:
                # This is an ensemble model
                models = []
                model_names = checkpoint['architecture']['models']
                
                for name in model_names:
                    if name == 'linear':
                        models.append(LinearRegressionNet(input_dim=input_dim, embed_dim=linear_embed_dim))
                    elif name == 'tree':
                        models.append(TreeBasedNet(input_dim=input_dim, embed_dim=linear_embed_dim))
                    elif name == 'gbm':
                        models.append(GBMNet(input_dim=input_dim, embed_dim=gbm_embed_dim))
                
                model = EnsembleModel(models, device)
                print(f"Created ensemble model with {len(models)} sub-models")
                
                # Load state dictionary
                try:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("Model state dictionary loaded successfully")
                except Exception as e:
                    print(f"Error loading state dict: {str(e)}")
                    print("Current model architecture:")
                    print(model)
                    print("\nCheckpoint info:")
                    for key, value in checkpoint.items():
                        print(f"{key}: {value}")
                    return False
                
                model.to(device)
                model.eval()
            else:
                print("Error: No model types specified in architecture")
                return False
        else:
            print("Error: Model architecture information missing from checkpoint")
            return False
        
        # Load normalization parameters
        if os.path.exists(norm_params_path):
            norm_params = joblib.load(norm_params_path)
            if 'scaler' in norm_params:
                scaler = norm_params['scaler']
                print("Loaded RobustScaler for data normalization")
        else:
            print("Normalization parameters not found. Using default normalization.")
            
        print(f"Model '{model_name}' loaded successfully")
        print(f"Required input dimension: {required_input_dim}")
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_mat_file(file_path):
    """Process MAT file with proper normalization"""
    try:
        # Load the MAT file
        mat_data = scipy.io.loadmat(file_path)
        
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
            print(f"Using array with key '{key}' from the MAT file")
        
        print("Original data shape:", data.shape)
        
        # Reshape data to match expected dimensions if necessary
        if data.shape[1] != required_input_dim:
            print(f"Reshaping data from {data.shape} to match required input dimension {required_input_dim}")
            
            # Flatten and tile/truncate to match required dimensions
            flattened = data.flatten()
            if len(flattened) >= required_input_dim:
                # If we have more data than needed, use the first required_input_dim elements
                reshaped_data = flattened[:required_input_dim].reshape(1, -1)
            else:
                # If we have less data than needed, tile the data to reach required size
                repeats = int(np.ceil(required_input_dim / len(flattened)))
                tiled_data = np.tile(flattened, repeats)
                reshaped_data = tiled_data[:required_input_dim].reshape(1, -1)
            
            data = reshaped_data
            print(f"Reshaped data shape: {data.shape}")
        
        # Apply normalization
        if scaler is not None:
            try:
                data = scaler.transform(data)
                print("Applied RobustScaler normalization")
            except Exception as e:
                print(f"Warning: Could not apply RobustScaler: {str(e)}")
                # Fallback to standard normalization
                data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                print("Applied fallback per-channel normalization")
        else:
            # Use per-channel normalization as fallback
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            print("Applied on-the-fly per-channel normalization")
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(data)
        print(f"Final tensor shape: {tensor_data.shape}")
        
        return tensor_data
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def make_prediction(data, threshold=0.5):
    """Make prediction with threshold"""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        probability = float(output.item())
        prediction = 1 if probability > threshold else 0
        return prediction, probability

# Load the model when app starts
if not load_model(model_name="linear_gbm_ensemble"):
    print("Warning: Failed to load model. API will not function correctly.")

# Flask routes
@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is not None:
        return jsonify({"status": "healthy", "model": "linear_gbm_ensemble"})
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if filename is valid
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .mat files are allowed"}), 400
    
    try:
        # Get threshold parameter
        threshold = float(request.form.get('threshold', 0.5))
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process file
        tensor_data = process_mat_file(file_path)
        
        # Delete the temporary file
        os.unlink(file_path)
        
        if tensor_data is None:
            return jsonify({"error": "Failed to process file"}), 400
        
        # Make prediction
        prediction, probability = make_prediction(tensor_data, threshold)
        
        # Return result
        return jsonify({
            "prediction": int(prediction),
            "probability": probability,
            "threshold": threshold,
            "prediction_label": "ADHD" if prediction == 1 else "No ADHD"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Default port
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting server on port {port}")
    print("Use CTRL+C to stop the server")
    
    # Use Waitress for production-grade serving
    serve(app, host='0.0.0.0', port=port)
