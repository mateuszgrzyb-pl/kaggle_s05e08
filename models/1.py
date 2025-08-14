# %% 1. Wczytanie bibliotek.
import numpy as np
from sklearnex.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from tools.base import read_data, split_data

# %% 2. Wczytanie zbioru.
train, test = read_data()
X_tr, X_va, y_tr, y_va = split_data(train, target='y')

# %% 3. Przygotowanie zbioru.
features_to_model = ['balance', 'age']

# %% 4. Modelowanie.
model = LogisticRegression(n_jobs=2)
skf = StratifiedKFold(n_splits=6)
cvs = cross_val_score(model,
                      cv=skf,
                      X=X_tr[features_to_model],
                      y=y_tr,
                      scoring='roc_auc')

model.fit(X_tr[features_to_model], y_tr)
pred_tr = model.predict_proba(X_tr[features_to_model])
pred_va = model.predict_proba(X_va[features_to_model])
pred_te = model.predict_proba(test[features_to_model])

score_tr = roc_auc_score(y_tr, pred_tr[:, 1])
score_va = roc_auc_score(y_va, pred_va[:, 1])

# wyniki
print('TR score: {}'.format(np.round(score_tr, 3)))  # 0.671
print('CV score: {}'.format(np.mean(cvs).round(3)))  # 0.67
