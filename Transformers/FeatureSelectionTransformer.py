from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        super().__init__()
        self.out_columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.out_columns]