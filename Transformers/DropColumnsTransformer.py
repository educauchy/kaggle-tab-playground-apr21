from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        super().__init__()
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X.drop(self.columns, axis=1, inplace=True)
        return self.X
