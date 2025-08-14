| Model File | Model Description | CV Score | Validation Score | Public Score | Training Date | Notes / Key Observations |
|------------|-----------------|---------|-----------------|--------------|---------------|-------------------------|
| `1.py`     | Baseline model using only 2 features: `'balance'` and `'age'`. | 0.670 | - | - | 2025-08-13 | Very simple baseline; serves as reference point for all other models. |
| `2.py`     | Single-feature model using `'duration'`. | 0.889 | - | - | 2025-08-13 | Demonstrates high predictive power of `'duration'` alone. |
| `3.py`     | "Banking" approach: all features except `'day'`. Classical preprocessing and modeling. | 0.944 | 0.944 | 0.94559 | 2025-08-14 | Strong classical model; consistent CV and validation performance. |
| `4.py`     | Basic PyTorch model: 2 linear layers, no activation, no weight initialization, no regularization, **trained locally** for only 3 epochs, small batch. | - | 0.9414 | 0.94357 | 2025-08-14 | Minimal neural network; competitive despite simplicity, shows potential of DL. |
| `5.py`     | Advanced PyTorch model: 3 linear layers, ReLU activations, 20% dropout, 30 epochs with early stopping. Trained on Kaggle notebook, using GPU. | - | 0.9622 | - | 2025-08-14 | Best deep learning configuration; early stopping prevented overfitting. |

