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
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="ADHD Model Bias Correction",
    page_icon="🧠",
    layout="wide"
)

# Helper functions for loading model and data
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
                return model
            except Exception:
                continue
                
        st.error("Could not load any model version")
        return None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_mat_file(file_content):
    """Process MAT file to ensure exact dimensions for model input"""
    try:
        # These must match EXACTLY what the model expects
        target_time_points = 1029
        target_features = 721
        expected_dim = 741570
        
        # Read and extract data
        bytes_data = io.BytesIO(file_content)
        mat_data = scipy.io.loadmat(bytes_data)
        data = mat_data['data'] if 'data' in mat_data else next(
            value for key, value in mat_data.items() 
            if not key.startswith('__') and isinstance(value, np.ndarray)
        )
        
        # Normalize the data
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        
        # Reshape data
        original_samples, n_channels = data.shape
        original_time = np.linspace(0, 1, original_samples)
        target_time = np.linspace(0, 1, target_time_points)
        
        final_data = np.zeros((1, expected_dim))
        n_channels_to_process = min(n_channels, target_features)
        
        for i in range(n_channels_to_process):
            interpolator = interp1d(original_time, data[:, i], kind='linear')
            interpolated_channel = interpolator(target_time)
            
            for t in range(target_time_points):
                final_data[0, i + (t * target_features)] = interpolated_channel[t]
        
        return torch.FloatTensor(final_data)
    except Exception as e:
        return None

# Calibration methods
class PlattScaling:
    """Probability calibration using Platt scaling (logistic regression)"""
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs')
        self.is_fitted = False
        
    def fit(self, scores, labels):
        # Reshape scores for sklearn
        self.model.fit(scores.reshape(-1, 1), labels)
        self.is_fitted = True
        
    def transform(self, scores):
        if not self.is_fitted:
            return scores
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]

class IsotonicCalibration:
    """Probability calibration using isotonic regression"""
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        
    def fit(self, scores, labels):
        self.model.fit(scores, labels)
        self.is_fitted = True
        
    def transform(self, scores):
        if not self.is_fitted:
            return scores
        return self.model.predict(scores)

class TemperatureScaling:
    """Simple temperature scaling for probability calibration"""
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        
    def fit(self, scores, labels):
        """Find optimal temperature via binary search"""
        # Simple optimization to find temperature
        best_loss = float('inf')
        best_temp = 1.0
        
        for temp in np.linspace(0.1, 10, 100):
            scaled = self._scale(scores, temp)
            # Binary cross entropy loss
            eps = 1e-15
            losses = -(labels * np.log(scaled + eps) + (1 - labels) * np.log(1 - scaled + eps))
            loss = np.mean(losses)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
                
        self.temperature = best_temp
        self.is_fitted = True
        return self
        
    def transform(self, scores):
        if not self.is_fitted:
            return scores
        return self._scale(scores, self.temperature)
    
    def _scale(self, scores, temperature):
        """Apply temperature scaling to logits"""
        # Convert probabilities to logits
        eps = 1e-15
        scores = np.clip(scores, eps, 1-eps)
        logits = np.log(scores / (1 - scores))
        
        # Scale logits by temperature
        scaled_logits = logits / temperature
        
        # Convert back to probabilities
        return 1 / (1 + np.exp(-scaled_logits))

def apply_calibration_to_file(file, model, calibrator):
    """Apply model prediction and calibration to a single file"""
    try:
        data = process_mat_file(file.getvalue())
        if data is None:
            return {'filename': file.name, 'status': 'Failed to process file'}
        
        # Get raw model prediction
        model.eval()
        with torch.no_grad():
            raw_prob = model(data).item()
            
        # Apply calibration if available
        if calibrator and calibrator.is_fitted:
            calib_prob = calibrator.transform(np.array([raw_prob]))[0]
        else:
            calib_prob = raw_prob
            
        return {
            'filename': file.name,
            'raw_probability': raw_prob,
            'calibrated_probability': calib_prob,
            'raw_prediction': 1 if raw_prob > 0.5 else 0,
            'calibrated_prediction': 1 if calib_prob > 0.5 else 0,
            'status': 'Processed'
        }
    except Exception as e:
        return {'filename': file.name, 'status': f'Error: {str(e)}'}

def plot_calibration_curve(probs, labels, calibrated_probs=None):
    """Plot calibration curve to visualize model calibration"""
    fig = go.Figure()
    
    # Calculate calibration curve for uncalibrated model
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='lines+markers',
        name='Original Model',
        line=dict(color='blue')
    ))
    
    # Calculate calibration curve for calibrated model if available
    if calibrated_probs is not None:
        calib_true, calib_pred = calibration_curve(labels, calibrated_probs, n_bins=10)
        fig.add_trace(go.Scatter(
            x=calib_pred, y=calib_true,
            mode='lines+markers',
            name='Calibrated Model',
            line=dict(color='green')
        ))
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        legend=dict(x=0, y=1.1, orientation='h'),
        width=700,
        height=500
    )
    
    return fig

def main():
    st.title("🧠 ADHD Model Bias Correction")
    st.write("""
    This app helps diagnose and correct bias in ADHD prediction models. You can:
    1. Analyze prediction bias with labeled test data
    2. Create a calibrated model to correct for bias
    3. Apply calibration to make more balanced predictions
    """)
    
    # Load the model
    model = load_model()
    if model is None:
        st.error("Failed to load model")
        st.stop()
        
    st.success("Model loaded successfully!")
    
    # Initialize session state for calibrator
    if 'calibrator' not in st.session_state:
        st.session_state.calibrator = None
        
    if 'calibration_data' not in st.session_state:
        st.session_state.calibration_data = None
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Model Diagnosis", "Create Calibration", "Apply Calibration"])
    
    with tab1:
        st.header("Model Bias Diagnosis")
        st.write("""
        Upload test files with known labels to analyze model bias.
        This will help determine if the model is systematically favoring one class.
        """)
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload labeled test files (.mat)", 
            type=['mat'], 
            accept_multiple_files=True,
            key="diagnosis_files"
        )
        
        # Label mapping
        st.subheader("Label Information")
        adhd_keyword = st.text_input("Keyword for ADHD files", "adhd")
        control_keyword = st.text_input("Keyword for control files", "control")
        
        if uploaded_files and st.button("Analyze Model Bias", key="analyze_bias"):
            results = []
            progress = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                # Determine true label from filename
                filename = file.name.lower()
                if adhd_keyword.lower() in filename:
                    true_label = 1  # ADHD
                elif control_keyword.lower() in filename:
                    true_label = 0  # Control
                else:
                    true_label = None
                    
                if true_label is not None:
                    # Process file and get prediction
                    try:
                        data = process_mat_file(file.getvalue())
                        if data is not None:
                            with torch.no_grad():
                                prob = model(data).item()
                                pred = 1 if prob > 0.5 else 0
                                
                            # Store result
                            results.append({
                                'filename': file.name,
                                'true_label': "ADHD" if true_label == 1 else "Control",
                                'probability': prob,
                                'prediction': "ADHD" if pred == 1 else "Control",
                                'correct': pred == true_label
                            })
                            
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                
                # Update progress
                progress.progress((i + 1) / len(uploaded_files))
                
            # Display results
            if results:
                df = pd.DataFrame(results)
                
                # Calculate key metrics
                adhd_files = df[df['true_label'] == "ADHD"]
                control_files = df[df['true_label'] == "Control"]
                
                adhd_accuracy = adhd_files['correct'].mean() if len(adhd_files) > 0 else 0
                control_accuracy = control_files['correct'].mean() if len(control_files) > 0 else 0
                
                adhd_bias = adhd_files['probability'].mean() if len(adhd_files) > 0 else 0
                control_bias = control_files['probability'].mean() if len(control_files) > 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Accuracy", f"{df['correct'].mean():.2f}")
                    
                with col2:
                    st.metric("ADHD Accuracy", f"{adhd_accuracy:.2f}")
                    st.metric("ADHD Avg. Probability", f"{adhd_bias:.2f}")
                    
                with col3:
                    st.metric("Control Accuracy", f"{control_accuracy:.2f}")
                    st.metric("Control Avg. Probability", f"{control_bias:.2f}")
                
                # Calculate bias metrics
                bias_score = adhd_bias - 0.75 if adhd_accuracy < 0.7 else 0
                bias_score += (1 - control_bias) - 0.75 if control_accuracy < 0.7 else 0
                
                # Bias interpretation
                if bias_score > 0.3:
                    st.error("⚠️ Significant model bias detected! Calibration recommended.")
                elif bias_score > 0.1:
                    st.warning("⚠️ Moderate model bias detected. Consider calibration.")
                else:
                    st.success("✓ Model appears to be reasonably well-calibrated.")
                
                # Show probability distribution
                fig = px.histogram(
                    df, x="probability", color="true_label",
                    title="Probability Distribution by True Label",
                    labels={"probability": "Predicted Probability", "count": "Count"}
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                st.plotly_chart(fig)
                
                # Save calibration data for later use
                calibration_data = {
                    'probabilities': df['probability'].values,
                    'labels': np.array([1 if label == "ADHD" else 0 for label in df['true_label']])
                }
                st.session_state.calibration_data = calibration_data
                
                # Display full results table
                st.subheader("Detailed Results")
                st.dataframe(df)
            else:
                st.error("No valid results obtained. Check if files have correct labels.")
    
    with tab2:
        st.header("Create Calibration Model")
        st.write("""
        Use labeled data to create a calibration model that corrects prediction bias.
        You must first run the Model Diagnosis tab to generate calibration data.
        """)
        
        if st.session_state.calibration_data is None:
            st.warning("No calibration data available. Please run the Model Diagnosis first.")
        else:
            # Calibration method selection
            calib_method = st.radio(
                "Select Calibration Method:",
                ["Platt Scaling (Logistic Regression)", 
                 "Isotonic Regression", 
                 "Temperature Scaling"]
            )
            
            if st.button("Create Calibration Model"):
                with st.spinner("Fitting calibration model..."):
                    # Get calibration data
                    probs = st.session_state.calibration_data['probabilities']
                    labels = st.session_state.calibration_data['labels']
                    
                    # Select and fit calibrator
                    if calib_method == "Platt Scaling (Logistic Regression)":
                        calibrator = PlattScaling()
                    elif calib_method == "Isotonic Regression":
                        calibrator = IsotonicCalibration()
                    else:
                        calibrator = TemperatureScaling()
                        
                    calibrator.fit(probs, labels)
                    st.session_state.calibrator = calibrator
                    
                    # Apply calibration to the same data
                    calibrated_probs = calibrator.transform(probs)
                    
                    # Show calibration curves
                    st.subheader("Calibration Results")
                    calib_fig = plot_calibration_curve(probs, labels, calibrated_probs)
                    st.plotly_chart(calib_fig)
                    
                    # Calculate improvement metrics
                    original_preds = (probs > 0.5).astype(int)
                    calibrated_preds = (calibrated_probs > 0.5).astype(int)
                    
                    original_accuracy = np.mean(original_preds == labels)
                    calibrated_accuracy = np.mean(calibrated_preds == labels)
                    
                    # Show metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Accuracy", f"{original_accuracy:.3f}")
                        
                    with col2:
                        st.metric("Calibrated Accuracy", f"{calibrated_accuracy:.3f}", 
                                 delta=f"{calibrated_accuracy-original_accuracy:.3f}")
                    
                    # Show samples before and after calibration
                    st.subheader("Sample Probability Adjustments")
                    
                    samples_df = pd.DataFrame({
                        'Original Probability': probs,
                        'Calibrated Probability': calibrated_probs,
                        'True Label': ["ADHD" if l == 1 else "Control" for l in labels],
                        'Original Prediction': ["ADHD" if p > 0.5 else "Control" for p in probs],
                        'Calibrated Prediction': ["ADHD" if p > 0.5 else "Control" for p in calibrated_probs],
                    })
                    
                    st.dataframe(samples_df)
                    
                    if calib_method == "Temperature Scaling":
                        st.info(f"Optimal temperature parameter: {calibrator.temperature:.3f}")
                        
                    st.success(f"Calibration model created successfully using {calib_method}!")
    
    with tab3:
        st.header("Apply Calibration to New Data")
        st.write("""
        Upload new files to make predictions with the calibrated model.
        """)
        
        # File uploader for prediction
        pred_files = st.file_uploader(
            "Upload files for prediction (.mat)", 
            type=['mat'], 
            accept_multiple_files=True,
            key="prediction_files"
        )
        
        if pred_files:
            if st.session_state.calibrator is None:
                st.warning("No calibration model available. Using original model predictions.")
                
            if st.button("Make Predictions"):
                pred_results = []
                progress = st.progress(0)
                
                for i, file in enumerate(pred_files):
                    # Apply prediction and calibration
                    result = apply_calibration_to_file(
                        file, model, st.session_state.calibrator
                    )
                    pred_results.append(result)
                    
                    # Update progress
                    progress.progress((i + 1) / len(pred_files))
                
                # Display prediction results
                if pred_results:
                    pred_df = pd.DataFrame(pred_results)
                    
                    # Display results summary
                    success_count = len(pred_df[pred_df['status'] == 'Processed'])
                    
                    st.success(f"Successfully processed {success_count} out of {len(pred_files)} files")
                    
                    # Show probability changes
                    if 'calibrated_probability' in pred_df.columns:
                        st.subheader("Raw vs Calibrated Probabilities")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=pred_df['raw_probability'], 
                            name='Raw Probabilities',
                            opacity=0.75
                        ))
                        fig.add_trace(go.Histogram(
                            x=pred_df['calibrated_probability'], 
                            name='Calibrated Probabilities',
                            opacity=0.75
                        ))
                        fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                        
                        fig.update_layout(
                            barmode='overlay',
                            title="Distribution of Probabilities",
                            xaxis_title="Probability",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig)
                    
                    # Show detailed results
                    st.subheader("Prediction Results")
                    st.dataframe(pred_df)
                    
                    # Allow download of results
                    csv = pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "calibrated_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.error("No valid prediction results available.")

if __name__ == "__main__":
    main()
