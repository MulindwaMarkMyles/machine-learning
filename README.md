## EEG-Based ADHD Classification Project

A deep learning approach to classify ADHD patterns from EEG signals using multi-strategy learning.

### 📋 Project Overview

- Multi-strategy Constrastive learning implementation for EEG signal processing
- Knowledge distillation with teacher-student architecture
- Real-time classification interface using Streamlit
- Supports both single and batch predictions

### 🛠️ Setup and Installation

#### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

#### Dependencies

```bash
pip install torch numpy pandas scipy scikit-learn streamlit joblib
```

#### Dataset Structure

```
Datasets Two/
├── ADHD_part2/
│   └── *.mat files
├── Control_part2/
│   └── *.mat files
└── models/
    ├── best_model_three.pth
    ├── model_config.pkl
    └── model_three_scaler.pkl
```

### 🚀 Training the Model

1. **Data Preparation**
    #### If you have cloned this repository skip this step.

   - Place ADHD .mat files in `ADHD_part2` directory
   - Place Control .mat files in `Control_part2` directory
   - Ensure all files follow the same format (19 channels)

2. **Configuration**

   - Adjust hyperparameters in `train_model.py` if needed
   - Default settings:
     - Learning rate: 0.001
     - Batch size: 32
     - Epochs: 100
     - Embedding dimensions: 128

3. **Start Training**

```bash
cd Model_Training
python train_model.py
```

4. **Monitor Progress**
   - Training metrics are displayed in real-time
   - Best model is saved automatically
   - Check `models/` directory for saved artifacts

### 💻 Running the Interface

1. **Launch the Application**

```bash
streamlit run interface.py
```

2. **Using the Interface**

   - Select input type (Single/Batch/CSV)
   - Upload your files
   - View predictions and confidence scores

3. **Supported File Formats**
   - Single .mat files
   - Multiple .mat files (batch processing)
   - CSV files (19 columns)

### 📊 Model Performance

- Training Accuracy: ~85%
- Validation Accuracy: ~82%
- Real-time prediction capability
- Robust to noise and variations in input

### 🔍 Troubleshooting

- Ensure .mat files contain 19-channel EEG data
- Check file permissions for models directory
- Verify CUDA availability for GPU training
- Monitor memory usage for large batch processing

### 📝 License

[Insert License Information]

### 👥 Contributors

[Insert Contributors Information]
