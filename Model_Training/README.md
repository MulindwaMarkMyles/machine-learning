# Model Optimization with Optuna

This directory contains scripts for optimizing the linear-GBM ensemble model using Optuna for hyperparameter tuning.

## Overview

The optimization process uses Optuna to find the best hyperparameters for the ensemble model consisting of Linear and GBM (Gradient Boosting Machine) models. The process includes:

1. Hyperparameter search for individual models
2. Optimization of ensemble weights
3. Cross-validation to ensure robustness
4. Final model training with best parameters

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Optimization

To start the optimization process:

```bash
python run_optimization.py --trials 50
```

Options:

- `--trials`: Number of optimization trials (default: 50)
- `--study-name`: Name of the optimization study (default: "ensemble_optimization")

## Optimization Parameters

The following hyperparameters are optimized:

- **Linear model**:

  - Embedding dimension
  - Learning rate
  - Weight decay

- **GBM model**:

  - Embedding dimension
  - Dropout rate
  - Learning rate
  - Weight decay

- **Ensemble**:
  - Training epochs for weight optimization
  - Learning rate for weight optimization
  - Batch size

## Results

The optimization results are saved in:

- `models/linear_gbm_ensemble_balanced_optimized.pth`: Optimized model weights
- `models/optimization_history.png`: Plot of optimization history
- `models/parameter_importances.png`: Plot of parameter importance

## Using the Optimized Model

The Flask application (`Flask_Deploy/app.py`) will automatically use the optimized model if available, falling back to the standard model if necessary.

## Monitoring

The optimization process creates a SQLite database (`ensemble_optimization.db`) that stores all trials. You can use Optuna's dashboard to visualize the optimization process:

```bash
optuna-dashboard sqlite:///ensemble_optimization.db
```
