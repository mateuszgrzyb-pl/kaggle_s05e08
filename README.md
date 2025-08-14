# Kaggle Playground Series S05E08
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

Experiments in applying advanced machine learning techniques — from traditional credit risk modeling to modern deep learning architectures — on tabular data from the [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s5e8) (Season 5, Episode 8).

## Overview
This project compares:
- **Traditional approaches**: Logistic Regression, Decision Tree, standard feature engineering.  
- **Deep Learning approaches**: PyTorch sequential networks with BatchNorm, Dropout, ReLU activations, and different architectures.

The goal is to evaluate whether deep learning can outperform well-established statistical methods in structured data problems, particularly in the financial sector.

## Features
- End-to-end preprocessing pipeline for tabular datasets (numerical scaling + one-hot encoding of categoricals).  
- Modular PyTorch training with early stopping, AUROC evaluation, and checkpointing.  
- Experimentation with different network sizes and regularization.  
- Output predictions ready for Kaggle submission.

## Project Structure
* `data/in` - raw input datasets
* `data/out` - model predictions
* `models` - scripts for modeling and EDA
* `tools/base` - data loading and splitting
* `tools/preprocessing` - preprocessing functions
* `tools/feature_selection` - feature selection methods

## Results
| Model File | Model Description | CV Score | Validation Score | Public Score | Training Date | Notes / Key Observations |
|------------|-----------------|---------|-----------------|--------------|---------------|-------------------------|
| `1.py`     | Baseline model using only 2 features: `'balance'` and `'age'`. | 0.670 | - | - | 2025-08-13 | Very simple baseline; serves as reference point for all other models. |
| `2.py`     | Single-feature model using `'duration'`. | 0.889 | - | - | 2025-08-13 | Demonstrates high predictive power of `'duration'` alone. |
| `3.py`     | "Banking" approach: all features except `'day'`. Classical preprocessing and modeling. | 0.944 | 0.944 | 0.94559 | 2025-08-14 | Strong classical model; consistent CV and validation performance. |
| `4.py`     | Basic PyTorch model: 2 linear layers, no activation, no weight initialization, no regularization, **trained locally** using CPU only, small batch. | - | 0.9414 | 0.94357 | 2025-08-14 | Minimal neural network; competitive despite simplicity, shows potential of DL. |
| `5.py`     | Advanced PyTorch model: 3 linear layers, ReLU activations, 20% dropout, 30 epochs with early stopping. Trained on a Kaggle notebook using GPU acceleration. | - | 0.9622 | - | 2025-08-14 | Best deep learning configuration; early stopping prevented overfitting. |


## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author
[Mateusz Grzyb](https://mateuszgrzyb.pl/o-mnie)
