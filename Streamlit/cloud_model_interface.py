import streamlit as st
import requests
import plotly.graph_objects as go
import os
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="ADHD Classification (Cloud Model)",
    page_icon="🧠",
    layout="wide"
)

# API settings
API_URL = os.environ.get("API_URL", "http://localhost:8000")
HEALTH_CHECK_ENDPOINT = f"{API_URL}/health"
PREDICT_ENDPOINT = f"{API_URL}/predict"

# Function to check API health
def check_api_health():
    try:
        response = requests.get(HEALTH_CHECK_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"API returned status code {response.status_code}"}
    except requests.RequestException as e:
        return False, {"error": f"Could not connect to API: {str(e)}"}

# Function to make prediction
def make_prediction(file_bytes, threshold=0.5):
    try:
        files = {"file": file_bytes}
        params = {"threshold": threshold}
        response = requests.post(PREDICT_ENDPOINT, files=files, params=params)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API error: {response.text}"}
    except requests.RequestException as e:
        return False, {"error": f"Request error: {str(e)}"}

# Function to display results
def display_results(results):
    prediction = results["prediction"]
    probability = results["probability"]
    threshold = results["threshold"]
    prediction_label = results["prediction_label"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display prediction with threshold context
        if prediction == 1:
            st.error(f"### Prediction: ADHD Indicated (threshold: {threshold:.2f})")
        else:
            st.success(f"### Prediction: No ADHD Indicated (threshold: {threshold:.2f})")
        
        # Display probability
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
    st.title("🧠 ADHD Classification System (Cloud Model)")
    st.write("""
    This application uses a cloud-deployed balanced model for ADHD classification.
    The model was specifically trained to avoid bias toward ADHD predictions.
    """)
    
    # Check API connection
    api_status, api_info = check_api_health()
    
    if api_status:
        st.success(f"✅ Connected to model API: {api_info.get('model', 'unknown model')}")
    else:
        st.error(f"❌ API connection problem: {api_info.get('error', 'unknown error')}")
        st.warning("Please verify the API_URL environment variable or check if the model server is running.")
        return
    
    # Sidebar for configuration
    st.sidebar.title("Model Settings")
    
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
    ### About the Model
    This cloud-based model is an ensemble of linear and gradient boosting models that was trained with special balancing techniques to reduce bias in ADHD classification.
    """)
    
    # File uploader
    st.subheader("Upload Brain Activity Data")
    
    uploaded_file = st.file_uploader("Choose a MAT file", type=['mat'])
    
    # Process file and make prediction
    if uploaded_file is not None:
        file_bytes = BytesIO(uploaded_file.getvalue())
        file_bytes.name = uploaded_file.name
        
        with st.spinner("Sending data to model API..."):
            success, results = make_prediction(file_bytes, threshold)
        
        if success:
            st.subheader("Prediction Results")
            display_results(results)
            
            # Additional visualization or explanation
            st.subheader("Understanding This Prediction")
            st.write("""
            The prediction above is based on a balanced model that was specifically trained 
            to avoid bias toward either ADHD or neurotypical classifications. 
            
            The threshold can be adjusted in the sidebar to make the model more or less sensitive
            to ADHD indicators in the data.
            """)
        else:
            st.error(f"Error: {results.get('error', 'Unknown error occurred')}")
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
