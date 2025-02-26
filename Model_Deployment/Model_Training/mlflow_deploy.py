import os
import json
import mlflow
import mlflow.pytorch
import torch
from fine_tune_models import ModelFineTuner
from multimodel_train import MATDataset

def setup_mlflow():
    """Setup MLflow tracking server"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ADHD_Classification")

def load_model_and_params(model_name):
    """Load model and its parameters"""
    with open('./models/fine_tuned_results.json', 'r') as f:
        results = json.load(f)
    
    if model_name not in results:
        raise ValueError(f"Model {model_name} not found in results")
    
    return results[model_name]

def register_model(model, model_name, params, metrics):
    """Register model with MLflow"""
    with mlflow.start_run(run_name=f"{model_name}_deployment"):
        # Log parameters
        for param_name, param_value in params['best_params'].items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['fine_tuned_score'])
        mlflow.log_metric("improvement", metrics['improvement'])
        
        # Log model
        mlflow.pytorch.log_model(
            model,
            f"{model_name}_model",
            registered_model_name=f"ADHD_{model_name.upper()}"
        )

def create_deployment_config(model_name):
    """Create deployment configuration"""
    return {
        "flavor": "pytorch",
        "target_uri": f"models:/{model_name}/Production",
        "config": {
            "instance_type": "ml.m4.xlarge",
            "instance_count": 1,
            "timeout_seconds": 60,
            "max_batch_size": 32
        }
    }

def deploy_model(model_name, device):
    """Deploy model to MLflow"""
    try:
        # Load model data
        model_data = load_model_and_params(model_name)
        
        # Setup dataset and create model
        dataset = MATDataset("../ADHD_part2/ADHD_part2", "../Control_part2/Control_part2")
        input_dim = dataset[0][0].shape[1] * dataset[0][0].shape[0]
        
        # Create model instance
        tuner = ModelFineTuner(dataset, device, model_name, model_data['best_params'])
        model = tuner.create_model(input_dim)
        
        # Load trained weights
        model_path = f'./models/{model_name}_optimized.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
        
        # Register model with MLflow
        register_model(model, model_name, model_data, model_data)
        
        # Create deployment configuration
        config = create_deployment_config(model_name)
        
        # Deploy model
        deployment = mlflow.deployments.get_deploy_client()
        deployment.create_endpoint(
            name=f"adhd_{model_name}_endpoint",
            config=config
        )
        
        print(f"Successfully deployed {model_name} model")
        return True
        
    except Exception as e:
        print(f"Error deploying {model_name} model: {str(e)}")
        return False

def main():
    # Setup MLflow
    setup_mlflow()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load available models
    with open('./models/fine_tuned_results.json', 'r') as f:
        models = json.load(f).keys()
    
    # Deploy each model
    successful_deployments = []
    failed_deployments = []
    
    for model_name in models:
        print(f"\nDeploying {model_name} model...")
        if deploy_model(model_name, device):
            successful_deployments.append(model_name)
        else:
            failed_deployments.append(model_name)
    
    # Print deployment summary
    print("\nDeployment Summary")
    print("=" * 50)
    print("\nSuccessfully deployed models:")
    for model in successful_deployments:
        print(f"- {model}")
    
    if failed_deployments:
        print("\nFailed deployments:")
        for model in failed_deployments:
            print(f"- {model}")

if __name__ == "__main__":
    main()
