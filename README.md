# wzy12410022@mail.dlut.edu.cn

# TDTR Multi-Task Learning: Interfacial Thermal Resistance & Liquid Effusivity Prediction

A PyTorch-based multi-task neural network for simultaneous prediction of three thermal transport properties from Time-Domain Thermoreflectance (TDTR) signals. The model takes experimental TDTR temperature decay curves and known material parameters as input, and predicts **ITR+** (ITR2 + ITR1), **ITR-** (ITR2 -ITR1), and **e_liquid** (liquid thermal effusivity).

## Overview

TDTR is a pump-probe optical technique widely used to measure thermal properties of thin films and interfaces. Traditional fitting of TDTR data to a thermal model is computationally expensive and sensitive to initial parameter guesses. This project replaces the iterative fitting process with a trained multi-head neural network that directly maps measured signals to physical properties.


## Model Architecture

```
Input (133 features)
  │
  ├── Encoder: Linear(133→256) + SiLU + BatchNorm1d
  │
  ├── ResBlock(256→256)
  ├── ResBlock(256→128)
  ├── ResBlock(128→64)
  │
  ├── Head 1: Linear(64→32→1)  → ITR+
  ├── Head 2: Linear(64→32→1)  → ITR-
  └── Head 3: Linear(64→32→1)  → e_liquid
```

Each residual block contains two fully-connected layers with SiLU activation, batch normalization, and a skip connection. The shared encoder learns a common representation of the TDTR signal, while three independent heads specialize for each prediction target.

## Input Features (133 total)

| Category | Count | Description |
|----------|-------|-------------|
| T1–T63 | 63 | Temperature signal (time-domain decay curve) |
| R1–R63 | 63 | Reflectance signal |
| SiO2_K | 1 | SiO₂ thermal conductivity |
| SiO2_cp | 1 | SiO₂ heat capacity |
| Al_d | 1 | Aluminum layer thickness (×10) |
| Al_K | 1 | Aluminum thermal conductivity (×100) |
| Al_cp | 1 | Aluminum heat capacity |
| Rpump | 1 | Pump beam radius (×10) |
| f | 1 | Modulation frequency |

## Project Structure

```
├── MTLCNN-R+-R-_e_liquid.ipynb   # Main notebook: training, evaluation, and inference
├── 2026-02-04.csv                 # Training dataset (30,000 samples, 136 columns)
├── kcl-2.csv                      # Example input for inference (2 samples)
├── models/
│   └── model_pure_data.pth        # Trained model weights (~1.2 MB)
├── scaler_x.pkl                   # Fitted StandardScaler for input normalization
├── kcl-2-test_Prediction_Direct.csv          # Direct prediction results
└── kcl-2-Final_Predictions_Only_Targets.csv  # MC uncertainty predictions
```

## Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy, Pandas, SciPy
- scikit-learn
- Matplotlib, Seaborn
- joblib
- openpyxl (optional, for Excel export)

Install with:

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn joblib openpyxl
```

## Quick Start

### 1. Training

Open `MTLCNN-R+-R-_e_liquid.ipynb` and run cells sequentially. Key configuration:

```python
config = {
    'seed': 5201314,
    'valid_ratio': 0.1,
    'n_epochs': 6000,
    'batch_size': 3000,
    'learning_rate': 1e-5,
    'early_stop': 500,
    'save_path': './models/model_pure_data.pth'
}
```

Training uses a weighted MSE loss (weights: ITR+ = 5.0, ITR- = 10.0, e_liquid = 5.0), Adam optimizer, gradient clipping (max_norm=1.0), and early stopping with patience of 500 epochs.

### 2. Inference on new data

Prepare a CSV with the same 133 feature columns as the training data, then run the inference cells. The model expects data to be normalized via the saved `scaler_x.pkl`.

```python
# Load model and scaler
scaler = joblib.load('scaler_x.pkl')
model = Advanced_TDTR_Net(input_dim=133)
model.load_state_dict(torch.load('./models/model_pure_data.pth'))
model.eval()

# Predict
x_scaled = scaler.transform(raw_data)
predictions = model(torch.from_numpy(x_scaled).float())
```

### 3. Uncertainty quantification (Monte Carlo)

The notebook includes an optional pipeline for MC-based uncertainty estimation: Gaussian noise is added to selected input parameters (SiO2_K, SiO2_cp, Al_d, Al_K, Al_cp, Rpump), the data is replicated 10,000×, and predictions are aggregated to obtain 95% confidence intervals via kernel density estimation (KDE).

## Training Details

- **Loss function**: Weighted MSE — the ITR- head receives double weight (10.0 vs 5.0) to compensate for its larger prediction difficulty
- **Regularization**: Dropout (p=0.05 during training, p=0.1 for inference), gradient clipping
- **Normalization**: StandardScaler fitted on training features, saved as `scaler_x.pkl`
- **Train/Validation split**: 90/10 stratified random split (seed fixed for reproducibility)

## Citation

If you use this code in your research, please consider citing the corresponding paper and linking back to this repository.

## License

This project is shared for academic and research purposes. Please contact the authors for other use cases.
