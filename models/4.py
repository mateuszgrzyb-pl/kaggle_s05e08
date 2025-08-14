# %% 1. Wczytanie bibliotek.
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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
        ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_features),
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
    nn.Linear(128, 1),
    nn.Sigmoid()
)

num_of_epoch = 3
criterion = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

# %% 5. Uczenie.
for epoch in range(num_of_epoch):
    print(f'Epocha numer {epoch+1}/{num_of_epoch}.')
    for data in train_dl:
        # 1. Zerowanie gradientów.
        optimizer.zero_grad()
        X, y = data
        # 2. Predykcja (forward pass) w celu wyznaczenia gradientów.
        pred = model(X)
        # 3. Wyznaczenie straty.
        loss = criterion(pred, y.reshape(-1, 1))
        # 4. Wyznaczenie gradientów.
        loss.backward()
        # 5. Aktualizacja parametrów modelu.
        optimizer.step()

# %% 6. Ewaluacja.
model.eval()
with torch.no_grad():
    all_preds = []
    all_targets = []
    for X, y in valid_dataset:
        pred = model(X)
        all_preds.append(pred)
        all_targets.append(y.reshape(-1, 1))

pred_va = torch.cat(all_preds)
y_va = torch.cat(all_targets)

auroc = AUROC(task='BINARY')
score_va = auroc(pred_va, y_va)  # 0.9414

# %% 7. Predykcja
with torch.no_grad():
    pred_te = model(torch.tensor(X_te))

pred_te = np.array(pred_te.tolist()).reshape(1, -1)[0]

pd.DataFrame({'id': test.index, 'y': pred_te}).to_csv('data/out/4.csv', index=False)
# [feature for feature in X_tr if feature not in features_to_analyze]
