import os
import json
import mlflow
import mlflow.pytorch
import torch
from ensemble_all import EnsembleModel
import shutil

def get_project_root():
    """Get absolute path to project root"""
    return os.path.abspath(os.path.dirname(__file__))

def setup_mlflow():
    """Setup MLflow with absolute path"""
    project_root = get_project_root()
    mlflow_db = os.path.join(project_root, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("ADHD_Classification_Ensemble")

def load_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_path = './models/best_ensemble_combination.pth'
    
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError("Best ensemble model file not found!")
    
    print("Loading pre-trained ensemble model...")
    checkpoint = torch.load(ensemble_path, map_location=device)
    
    # Load the complete ensemble model
    ensemble = checkpoint['ensemble']
    ensemble.eval()
    
    print("Best ensemble model loaded successfully!")
    print(f"Model names: {checkpoint['model_names']}")
    print(f"Metrics: {checkpoint['metrics']}")
    
    return ensemble

def create_deployment_info(model_uri):
    """Create deployment information including endpoints"""
    deployment_info = {
        "model_uri": model_uri,
        "endpoint": {
            "name": "adhd_ensemble_endpoint",
            "url": mlflow.get_tracking_uri() + "/api/2.0/preview/mlflow/model-versions/get-download-uri",
            "model_name": "ADHD_BEST_ENSEMBLE",
            "version": "1"
        },
        "serving_config": {
            "max_batch_size": 16,
            "timeout_seconds": 60
        }
    }
    
    # Save deployment info to JSON
    os.makedirs('./models', exist_ok=True)
    with open('./models/ensemble_deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=4)
    
    return deployment_info

def main():
    setup_mlflow()
    
    print("Loading best ensemble model...")
    ensemble = load_ensemble()
    
    print("Registering model with MLflow...")
    with mlflow.start_run(run_name="best_ensemble_deployment") as run:
        # Create artifacts directory
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save model code
        shutil.copy2("ensemble_all.py", os.path.join(artifacts_dir, "ensemble_all.py"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "ensemble_all.py"), "code")
        
        # Log the model
        model_info = mlflow.pytorch.log_model(
            ensemble,
            "best_ensemble",
            registered_model_name="ADHD_BEST_ENSEMBLE"
        )
        
        # Create and save deployment info
        print("Saving deployment information...")
        deployment_info = create_deployment_info(model_info.model_uri)
        
        # Save deployment info as artifact
        with open(os.path.join(artifacts_dir, "deployment_info.json"), "w") as f:
            json.dump(deployment_info, f, indent=4)
        mlflow.log_artifact(os.path.join(artifacts_dir, "deployment_info.json"), "deployment")
        
        print("\nDeployment Info Summary:")
        print(f"Model URI: {deployment_info['model_uri']}")
        print(f"Endpoint Name: {deployment_info['endpoint']['name']}")
        print(f"Endpoint URL: {deployment_info['endpoint']['url']}")
    
    # Clean up artifacts directory
    shutil.rmtree(artifacts_dir)
    
    print("\nBest ensemble model successfully registered and deployment info saved!")

if __name__ == "__main__":
    main()
