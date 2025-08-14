import copy
import tqdm
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def bin_data(Xs,
             y_tr,
             features_to_bin,
             min_samples_leaf=0.05,
             max_depth=3,
             random_state=42,
             regression=False,
             verbose=False):
    """
    Binning cech za pomocą drzew decyzyjnych.

    Parameters
    ----------
    Xs : list of pandas.DataFrame
        Lista zestawów danych wejściowych zawierających zmienne niezależne.
    y_tr : pandas.Series lub numpy.array
        Zmienna zależna (target) używana do dopasowania modelu.
    features_to_bin : list
        Lista cech (kolumn), które mają być przetworzone (binned).
    min_samples_leaf : int lub float, optional
        Minimalna liczba próbek, które muszą znaleźć się w liściu. Jeśli wartość jest
        w zakresie (0, 1), traktowana jest jako procent próbek. Domyślnie 0.05.
    max_depth : int, optional
        Maksymalna głębokość drzewa decyzyjnego. Domyślnie 3.
    random_state : int, optional
        Losowy stan używany do kontrolowania losowości drzewa decyzyjnego. Domyślnie 42.
    regression : bool, optional
        Flaga wskazująca, czy użyć regresora (True) czy klasyfikatora (False). Domyślnie False.
    verbose : bool, optional
        Flaga włączająca pasek postępu (True) lub wyłączająca (False). Domyślnie False.

    Returns
    -------
    Xs : list of pandas.DataFrame
        Lista przetworzonych zestawów danych z binned cechami.

    """
    Xs = copy.copy(Xs)

    if regression:
        estimator = DecisionTreeRegressor(max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          random_state=random_state)
    else:
        estimator = DecisionTreeClassifier(max_depth=max_depth,
                                           min_samples_leaf=min_samples_leaf,
                                           random_state=random_state)
    for feature in tqdm.tqdm(features_to_bin, disable=(not verbose)):
        estimator.fit(Xs[0][[feature]], y_tr)
        for X in Xs:
            X.loc[:, feature] = estimator.apply(X[[feature]]).astype(float)
    return Xs


def generate_interactions(X, degree=2):
    """
    Funkcja do budowy interakcji.

    Parameters
    ----------
    X : pandas.DataFrame
        Dane wejściowe zawierające zmienne niezależne.
    degree : int, optional
        Maksymalny stopień interakcji. Domyślnie 2.

    Returns
    -------
    X_inte : pandas.DataFrame
        Dane wejściowe z polynomial features oraz odpowiednimi nazwami kolumn.

    """
    poly = PolynomialFeatures(degree=degree+1, interaction_only=True, include_bias=False)
    X_inte = poly.fit_transform(X)
    # Generowanie nazw kolumn
    original_features = X.columns
    inte_feature_names = poly.get_feature_names_out(original_features)

    X_inte_df = pd.DataFrame(X_inte, columns=inte_feature_names)

    return X_inte_df
