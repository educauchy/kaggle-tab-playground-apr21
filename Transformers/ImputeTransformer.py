from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import pandas as pd


class ImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type='KNN', **params):
        super().__init__()
        self.type = type
        self.imputers = {
            'KNN': KNNImputer,
            'iterative': IterativeImputer,
        }
        self.imputer = self.imputers[type](**params)

    def fit(self, X, y=None):
        print(self.imputer)
        print('Imputing begins...')
        self.imputer.fit(X)
        print('Imputing ended...')
        print('')
        return self

    def transform(self, X, y=None):
        columns = X.columns
        imputed_data = self.imputer.transform(X)
        return pd.DataFrame(imputed_data, columns=columns)
