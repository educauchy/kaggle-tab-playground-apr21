from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


class EncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type='label', column='', out_column='', data=None):
        super().__init__()
        self.type = type
        self.column = column
        self.out_column = out_column
        self.data = data
        self.encoders = {
            'label': LabelEncoder(),
            'ordinal': OrdinalEncoder(),
            'onehot': OneHotEncoder()
        }
        self.encoder = self.encoders[type]

    def fit(self, X, y=None):
        if self.data is None:
            self.encoder.fit(X[self.column])
        else:
            self.encoder.fit(self.data)
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X[self.out_column] = self.encoder.transform(X[self.column])
        return self.X