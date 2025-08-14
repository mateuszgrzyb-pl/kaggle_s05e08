# %% 1. Wczytanie bibliotek.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import AUROC

from tools.base import read_data, split_data

# %% 2. Wczytanie zbioru.
train, test = read_data()
X_tr, X_va, y_tr, y_va = split_data(train, target='y')

# %% 3. Przygotowanie zbioru.
numerical_features = ['balance', 'age', 'duration', 'previous', 'pdays', 'campaign']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
features_to_analyze = numerical_features + categorical_features

# 3.1. Konwersja zmiennych.
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("norm", StandardScaler(), numerical_features)
    ],
    remainder="passthrough"  # reszta kolumn bez zmian
)

preprocessor.fit(X_tr)
X_tr = preprocessor.transform(X_tr)
X_va = preprocessor.transform(X_va)
X_te = preprocessor.transform(test)

# 3.2. Konwersja zbioru do PyTorch.
train_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr.to_numpy(), dtype=torch.float32))
valid_dataset = TensorDataset(torch.tensor(X_va), torch.tensor(y_va.to_numpy(), dtype=torch.float32))

train_dl = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_dl = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=True)

# %% 4. Przygotowanie modelu.
model = nn.Sequential(
    nn.Linear(51, 128),
    nn.BatchNorm1d(128),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-5)

# %% 5. Uczenie.
num_of_epoch = 30
best_score = -np.inf 
patience = 5
epochs_no_improve = 0

for epoch in range(num_of_epoch):
    print(f'Epocha numer {epoch+1}/{num_of_epoch}.')
    model.train()
    for X, y in train_dl:
        X = X.to(device)
        y = y.to(device).reshape(-1,1)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in valid_dl:
            X = X.to(device)
            y = y.to(device).reshape(-1,1)
            pred = model(X)
            all_preds.append(pred)
            all_targets.append(y)
    pred_va = torch.cat(all_preds)
    y_va = torch.cat(all_targets)
    auroc = AUROC(task='BINARY').to(device)
    score_va = auroc(pred_va, y_va)
    scheduler.step(score_va)
    print('VA score:', round(score_va.item(),4))

    if score_va > best_score:
        best_score = score_va.item()
        epochs_no_improve = 0
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"Nowy najlepszy wynik! Licznik cierpliwości zresetowany.")
    else:
        epochs_no_improve += 1
        print(f"Brak poprawy. Licznik cierpliwości: {epochs_no_improve}/{patience}")
    if epochs_no_improve >= patience:
        print(f"Przerwanie treningu. Brak poprawy przez {patience} epok.")
        break

print(f"\nTrening zakończony. Najlepszy wynik walidacyjny (VA score): {best_score:.4f}")

# %% 7. Predykcja
with torch.no_grad():
    pred_te = model(torch.tensor(X_te))

pred_te = np.array(pred_te.tolist()).reshape(1, -1)[0]

pd.DataFrame({'id': test.index, 'y': pred_te}).to_csv('data/out/5.csv', index=False)

