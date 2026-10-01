# Remaining Useful Life Prediction with 1D CNN

A deep learning approach for predicting the Remaining Useful Life (RUL) of turbofan engines from multivariate time-series sensor data using a one-dimensional convolutional neural network (1D CNN).

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset to model engine degradation and estimate the number of operational cycles remaining before failure.

## Overview

Remaining Useful Life prediction is an important problem in predictive maintenance. Accurate RUL estimates can help identify equipment approaching failure, support maintenance planning, and reduce unnecessary component replacement.

This project implements an end-to-end RUL prediction pipeline that:

- Loads and preprocesses multivariate turbofan sensor data
- Removes near-constant sensor features
- Normalizes features using training-set statistics
- Generates fixed-length temporal windows from engine trajectories
- Splits engines into independent training and validation sets
- Applies Gaussian noise augmentation to training sequences
- Trains a 1D CNN for RUL regression
- Uses mixed-precision training when GPU acceleration is available
- Evaluates predictions using RMSE, MAE, and tolerance-based accuracy

## Dataset

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset.

C-MAPSS contains run-to-failure trajectories for simulated turbofan engines. Each trajectory consists of operating conditions and multivariate sensor measurements collected over successive operating cycles.

The current experiment focuses on the **FD001** subset.

Training trajectories contain complete engine lifecycles through failure, allowing the RUL target at each timestep to be calculated as:

RUL = final engine cycle - current cycle

For test engines, only partial trajectories are provided, and the model predicts the remaining useful life from the final observed sensor sequence.

## Model Architecture

The model is a 1D convolutional neural network designed to extract temporal degradation patterns from multivariate sensor sequences.

The architecture includes:

- Three convolutional blocks
- Batch normalization
- ReLU activations
- Dilated 1D convolutions
- Max pooling
- Dropout regularization
- Adaptive average pooling
- Linear regression output

Sensor measurements are converted into sliding windows before being passed to the network, allowing the CNN to learn temporal relationships across consecutive operating cycles.

Training uses AdamW optimization, Huber loss, learning-rate scheduling, Gaussian noise augmentation, and mixed-precision computation when CUDA is available.

## Results

Evaluation on the FD001 test set produced:

| Metric | Result |
| --- | ---: |
| RMSE | 24.35 cycles |
| MAE | 16.32 cycles |
| Predictions within ±10 cycles | 49% |
| Predictions within ±20 cycles | 68% |
| Predictions within ±30 cycles | 84% |
| Predictions within ±40 cycles | 90% |
| Predictions within ±50 cycles | 97% |

The model successfully learned temporal degradation patterns from the sensor sequences and achieved substantially lower training and validation errors over the course of training.

The difference between validation and test performance indicates a remaining generalization gap, suggesting opportunities for additional hyperparameter tuning, alternative sequence architectures, and evaluation across the more complex C-MAPSS subsets.

## Project Structure

```text
.
├── data/
│   └── ...
├── rul_cnn_analysis.ipynb
├── rul_prediction_cnn.py
├── img1.png
├── README.md
└── requirements.txt