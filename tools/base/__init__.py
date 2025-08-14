import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin


def read_data():
    train = pd.read_csv('data/in/train.csv', index_col=0)
    test = pd.read_csv('data/in/test.csv', index_col=0)
    return train, test


def split_data(train, target):
    train, valid = train_test_split(train, test_size=0.3, random_state=42, stratify=train[target])
    y_tr = train[target].copy()
    X_tr = train.drop(columns=[target]).copy()
    y_va = valid[target].copy()
    X_va = valid.drop(columns=[target]).copy()
    return X_tr, X_va, y_tr, y_va


class SMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, a=100, b=False):
        self.a = a
        self.b = b
        self.model_class = model_class

    def fit(self, X, y):
        self.model_ = self.model_class(max_iter=self.a, scale=self.b)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
