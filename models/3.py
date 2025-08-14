# %% 1. Wczytanie bibliotek.
import numpy as np
import pandas as pd
from sklearnex.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from category_encoders import WOEEncoder
from mlxtend.feature_selection import SequentialFeatureSelector

from tools.base import read_data, split_data
from tools.feature_selection import one_dim_analysis, correlation_features_selection
from tools.preprocessing import bin_data

# %% 2. Wczytanie zbioru.
train, test = read_data()
X_tr, X_va, y_tr, y_va = split_data(train, target='y')

# %% 3. Przygotowanie zbioru.
numerical_features = ['balance', 'age', 'duration', 'previous', 'pdays', 'campaign']
categorical_feature = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
features_to_analyze = numerical_features + categorical_feature

# 3.1. Kategoryzacja zmiennych ciągłych
X_tr, X_va, test = bin_data([X_tr, X_va, test],
                            y_tr,
                            numerical_features,
                            min_samples_leaf=0.05,
                            max_depth=3,
                            random_state=2001)

# 3.2. Zmiana kodowania na WoE
woe = WOEEncoder(cols=features_to_analyze, random_state=2001)
woe.fit(X_tr, y_tr)
X_tr = woe.transform(X_tr)
X_va = woe.transform(X_va)
test = woe.transform(test)

# %% 4. Analiza jednowymiarowa.
model = LogisticRegression(n_jobs=2)
skf = StratifiedKFold(n_splits=6)
oda = one_dim_analysis(X_tr, y_tr, features_to_analyze, model, scoring='roc_auc', cv_=skf)

# duration     0.886742
# balance      0.681629
# month        0.653227
# job          0.622133
# housing      0.617575
# contact      0.616225
# age          0.596824
# poutcome     0.585183
# pdays        0.583383
# previous     0.579423
# campaign     0.577915
# education    0.570409
# marital      0.564175
# loan         0.543525
# default      0.505974

features_to_model = oda[oda > 0.55].index.tolist()
oda = oda.loc[features_to_model]

# %% 5. Selekcja korelacyjna.
cfs = correlation_features_selection(X_tr, oda)

# %% 6. Selekcja krokowa.
model = LogisticRegression(n_jobs=2)
skf = StratifiedKFold(n_splits=6)
sfs = SequentialFeatureSelector(model, k_features=(3, 11), scoring='roc_auc', cv=skf, verbose=10)
sfs.fit(X_tr[cfs], y_tr)
features_to_model = list(sfs.k_feature_names_)
len(features_to_model)
# %% 7. Modelowanie.
model = LogisticRegression(n_jobs=2)
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
print('TR score: {}'.format(np.round(score_tr, 3)))  # 0.944
print('VA score: {}'.format(np.round(score_va, 3)))  # 0.944
print('CV score: {}'.format(np.mean(cvs).round(3)))  # 0.944

pd.DataFrame({'id': test.index, 'y': pred_te[:, 1]}).to_csv('data/out/3.csv', index=False)
# [feature for feature in X_tr if feature not in features_to_analyze]
