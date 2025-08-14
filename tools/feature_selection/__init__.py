import tqdm
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.model_selection import cross_val_score


def one_dim_analysis(X, y, features_to_check, estimator, scoring=None, cv_=5, n_jobs=-2):
    """
    Analiza jednowymiarowa.

    Parameters
    ----------
    X : pandas.DataFrame
        Dane wejściowe zawierające zmienne niezależne.
    y : pandas.Series lub numpy.array
        Zmienna zależna, którą chcemy przewidywać.
    features_to_check : list
        Lista cech (kolumn) do sprawdzenia w analizie jednowymiarowej.
    estimator : estimator object
        Model (klasyfikator lub regresor) do użycia w analizie jednowymiarowej.
    scoring : str, optional
        Metryka oceny używana podczas walidacji krzyżowej. Domyślnie None, co oznacza użycie
        'neg_root_mean_squared_error' dla regresji lub 'roc_auc' dla klasyfikacji.
    cv_ : int, optional
        Liczba podziałów danych do walidacji krzyżowej. Domyślnie 5.
    n_jobs : int, optional
        Liczba równoległych prac do uruchomienia podczas walidacji krzyżowej. Domyślnie -2, co oznacza
        użycie wszystkich dostępnych rdzeni procesora minus jeden.

    Returns
    -------
    results : pandas.Series
        Posortowana seria wyników walidacji krzyżowej dla każdej cechy.

    """
    # Ustawienie domyślnej metryki na podstawie typu modelu
    if scoring is None:
        if is_classifier(estimator):
            scoring = 'roc_auc'
        else:
            scoring = 'neg_root_mean_squared_error'

    results = []
    for feature in tqdm.tqdm(features_to_check):
        cv = cross_val_score(estimator,
                             X[[feature]],
                             y,
                             scoring=scoring,
                             cv=cv_,
                             n_jobs=n_jobs)
        results.append([feature, np.mean(cv), np.std(cv)])

    results = np.array(results)
    results = pd.Series(data=results[:, 1].astype(float), index=results[:, 0])
    results.sort_values(ascending=False, inplace=True)
    return results


def one_dim_analysis_penalized(X, y, features_to_check, estimator, scoring=None, cv_=5, n_jobs=-2):
    """
    Analiza jednowymiarowa.

    Parameters
    ----------
    X : pandas.DataFrame
        Dane wejściowe zawierające zmienne niezależne.
    y : pandas.Series lub numpy.array
        Zmienna zależna, którą chcemy przewidywać.
    features_to_check : list
        Lista cech (kolumn) do sprawdzenia w analizie jednowymiarowej.
    estimator : estimator object
        Model (klasyfikator lub regresor) do użycia w analizie jednowymiarowej.
    scoring : str, optional
        Metryka oceny używana podczas walidacji krzyżowej. Domyślnie None, co oznacza użycie
        'neg_root_mean_squared_error' dla regresji lub 'roc_auc' dla klasyfikacji.
    cv_ : int, optional
        Liczba podziałów danych do walidacji krzyżowej. Domyślnie 5.
    n_jobs : int, optional
        Liczba równoległych prac do uruchomienia podczas walidacji krzyżowej. Domyślnie -2, co oznacza
        użycie wszystkich dostępnych rdzeni procesora minus jeden.

    Returns
    -------
    results : pandas.Series
        Posortowana seria wyników walidacji krzyżowej dla każdej cechy.

    """
    # Ustawienie domyślnej metryki na podstawie typu modelu
    if scoring is None:
        if is_classifier(estimator):
            scoring = 'roc_auc'
        else:
            scoring = 'neg_root_mean_squared_error'

    results = []
    for feature in tqdm.tqdm(features_to_check):
        cv = cross_val_score(estimator,
                             X[[feature]],
                             y,
                             scoring=scoring,
                             cv=cv_,
                             n_jobs=n_jobs)
        results.append([feature, np.mean(cv), np.std(cv)])

    results = np.array(results)
    results = pd.Series(data=results[:, 1].astype(float)-results[:, 2].astype(float), index=results[:, 0])
    results.sort_values(ascending=False, inplace=True)
    return results


def correlation_features_selection(X,
                                   one_dim_results,
                                   corr_level=0.5,
                                   method='pearson'):
    """
    Selekcja cech na podstawie korelacji.

    Parameters
    ----------
    X : pandas.DataFrame
        Dane wejściowe zawierające zmienne niezależne.
    one_dim_results : pandas.Series
        Wyniki analizy jednowymiarowej zawierające oceny cech.
    corr_level : float, optional
        Poziom korelacji, powyżej którego cechy są odrzucane. Domyślnie 0.5.
    method : str, optional
        Metoda obliczania korelacji ('pearson', 'spearman', 'kendall'). Domyślnie 'pearson'.

    Returns
    -------
    selected_features : list
        Lista wybranych cech, które przeszły próg korelacji.

    """
    features_to_check = one_dim_results.index.tolist()
    selected_features = []

    progress_bar = tqdm.tqdm(total=len(features_to_check))

    while len(features_to_check) >= 1:
        feature = features_to_check.pop(0)
        selected_features.append(feature)
        progress_bar.update(1)  # Zaktualizuj pasek postępu o jedną iterację
        corr_tab = X[features_to_check]\
            .corrwith(X[feature], method=method)\
            .abs()\
            .sort_values(ascending=False)
        features_to_drop = corr_tab[corr_tab > corr_level].index.tolist()
        for feature in features_to_drop:
            features_to_check.remove(feature)
        progress_bar.update(len(features_to_drop))  # Zaktualizuj pasek postępu o liczbę usuniętych cech
    progress_bar.close()  # Zamknij pasek postępu po zakończeniu
    return selected_features
