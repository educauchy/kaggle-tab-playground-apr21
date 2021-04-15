from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
import pandas as pd


class ImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type='KNN', n_neighbors=5):
        super().__init__()
        self.imputers = {
            'KNN': KNNImputer(n_neighbors=n_neighbors),
            'iterative': IterativeImputer(),
        }
        self.imputer = self.imputers[type]

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('Imputing begins...')
        columns = X.columns
        imputed_data = self.imputer.transform(X)
        print('Imputing ended...')
        print('')
        return pd.DataFrame(imputed_data, columns=columns)