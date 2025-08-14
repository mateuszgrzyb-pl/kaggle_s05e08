# %% 1. Wczytanie bibliotek.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from pygam import LogisticGAM, s
from tools.base import read_data, split_data
from tools.feature_selection import one_dim_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.special import expit

# %% 2. Wczytanie zbioru.
train, test = read_data()
X_tr, X_va, y_tr, y_va = split_data(train, target='y')

# %% 3. Analiza jednowymiarowa
numeric_features = ['balance', 'age', 'duration', 'previous', 'pdays', 'campaign']
logit = LogisticRegression(n_jobs=2)
skf = StratifiedKFold(n_splits=6)
oda = one_dim_analysis(X_tr, y_tr, numeric_features, logit, scoring='roc_auc', cv_=skf)
# duration    0.888988
# balance     0.674970
# previous    0.580233
# campaign    0.578349
# pdays       0.575100
# age         0.480307

dt = DecisionTreeClassifier(max_depth=3, min_samples_split=0.1)
oda = one_dim_analysis(X_tr, y_tr, numeric_features, dt, scoring='roc_auc', cv_=skf)
# duration    0.886458
# balance     0.678754
# age         0.584575
# pdays       0.583502
# previous    0.580226
# campaign    0.577891

# Wnioski:
# age - nieliniowa zależność

# %% 4. Wizualizacja danych - analiza 2D.
# 4.1. Wiek.
X_tr.columns
X_tr.month.unique()
sns.barplot(train, x='default', y='y')
plt.show()

