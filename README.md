# VAE for Anomaly Detection

A PyTorch implementation of Variational Autoencoders (VAEs) for industrial anomaly detection and root cause analysis.

## Overview

This project demonstrates using VAEs as a "digital twin" for factory sensor monitoring. The VAE learns the normal relationships between sensors and flags anomalies when those relationships break down—even before individual sensors hit alarm thresholds.

### Key Features

- **Unsupervised Learning**: Train only on "healthy" data—no labeled anomalies required
- **Anomaly Detection**: Flag abnormal sensor patterns via reconstruction error
- **Root Cause Analysis**: Identify which specific sensors/components are failing
- **Latent Space Visualization**: See how data clusters and drifts over time
- **Predictive Maintenance**: Detect degradation before complete failure

## Installation

```bash
# Clone or navigate to project directory
cd "VAE for Anomaly Detection"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook experiment.ipynb
```

The notebook provides an interactive walkthrough of the entire pipeline.

### Option 2: Python Script

```python
from src.data_loader import BearingDataLoader
from src.vae_model import VAE
from src.trainer import VAETrainer
from src.anomaly_detector import AnomalyDetector

# Load data (uses synthetic bearing data by default)
loader = BearingDataLoader(window_size=2048, stride=1024)
X_train, X_test, y_train, y_test = loader.load_data(use_synthetic=True)

# Build model
model = VAE(
    input_dim=X_train.shape[1],
    hidden_dims=[512, 256, 128],
    latent_dim=16
)

# Train on healthy data only
trainer = VAETrainer(model)
history = trainer.train(X_train, epochs=100)

# Detect anomalies
detector = AnomalyDetector(model)
detector.fit_threshold(X_train, method='percentile', percentile=99)
metrics = detector.evaluate(X_test, y_test)

# Root cause analysis
anomalies = X_test[y_test == 1]
root_causes = detector.analyze_root_cause(anomalies, top_k=10)
```

## Project Structure

```
VAE for Anomaly Detection/
├── src/
│   ├── __init__.py          # Package exports
│   ├── data_loader.py       # Data loading and windowing utilities
│   ├── vae_model.py         # VAE architectures (Dense and Conv1D)
│   ├── trainer.py           # Training loop with early stopping
│   ├── anomaly_detector.py  # Anomaly detection and root cause analysis
│   └── visualization.py     # Plotting utilities
├── experiment.ipynb         # Interactive notebook tutorial
├── requirements.txt         # Python dependencies
└── README.md
```

## How It Works

### 1. Data Preparation

Sensor time-series data is windowed into fixed-length segments:

```
Raw: [s1, s2, s3, s4, s5, s6, s7, s8, ...]
         ↓ window_size=4, stride=2
Windows: [[s1,s2,s3,s4], [s3,s4,s5,s6], [s5,s6,s7,s8], ...]
```

### 2. VAE Training

The VAE learns to compress and reconstruct "normal" sensor patterns:

```
Input (sensors) → Encoder → Latent Space (μ, σ) → Decoder → Reconstruction
                              ↑
                    Learned "normal" representation
```

### 3. Anomaly Detection

When new data arrives:
- **Low reconstruction error** → Normal operation
- **High reconstruction error** → Anomaly detected

### 4. Root Cause Analysis

For flagged anomalies, examine per-feature reconstruction error:
- The sensor with highest error is the likely root cause
- Aggregating errors by component (e.g., bearing) localizes the fault

## Configuration Options

### VAE Model

```python
VAE(
    input_dim=8192,           # Input feature dimension
    hidden_dims=[512, 256],   # Encoder layer sizes
    latent_dim=16,            # Latent space dimension
    dropout=0.1,              # Regularization
    beta=1.0                  # β-VAE weight (higher = more regularized)
)
```

### Data Loader

```python
BearingDataLoader(
    window_size=2048,         # Samples per window
    stride=1024,              # Window step (overlap)
    use_frequency_features=False  # FFT features vs raw time-series
)
```

### Anomaly Detector

```python
detector.fit_threshold(
    X_train,
    method='percentile',      # 'percentile' or 'std'
    percentile=99             # Threshold percentile
)
```

## Using Real Data

### NASA Bearing Dataset

1. Download from [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
2. Extract to `data/nasa_bearing/`
3. Set `use_synthetic=False` in the data loader

### Custom Data

Implement a custom loader following the `BearingDataLoader` interface:
- Return `(X_train, X_test, y_train, y_test)`
- `X_train` should contain only healthy samples
- `y_test` should have labels (0=normal, 1=anomaly)

## Results Interpretation

### Metrics

- **AUC-ROC**: Area under ROC curve (0.5=random, 1.0=perfect)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Of predicted anomalies, how many are real?
- **Recall**: Of real anomalies, how many were detected?

### Latent Space

- Normal data clusters near the origin
- Anomalies appear as outliers
- Drift toward outlier region indicates impending failure

### Root Cause

The feature/sensor with highest reconstruction error contribution is the likely fault source.

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013
- [β-VAE: Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fzU9gl) - Higgins et al., 2017
- [Variational Autoencoders for Anomaly Detection](https://arxiv.org/abs/1802.03903)
