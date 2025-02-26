import os
import argparse
from optuna_ensemble_tuner import main as run_optimization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna-based model optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, default="ensemble_optimization", help="Name of the optimization study")
    args = parser.parse_args()
    
    print(f"Starting optimization with {args.trials} trials...")
    
    # Set environment variables
    os.environ["N_TRIALS"] = str(args.trials)
    os.environ["STUDY_NAME"] = args.study_name
    
    # Run the optimization
    run_optimization()
