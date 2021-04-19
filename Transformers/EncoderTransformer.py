from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import numpy as np


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
        self.X = X.copy()
        if self.data is None:
            self.encoder.fit(self.X[self.column].astype(str))
        else:
            self.encoder.fit(self.data.astype(str))
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X[self.out_column] = self.encoder.transform(self.X[self.column].astype(str))
        self.X.loc[self.X[self.column].isnull(), self.out_column] = np.nan
        return self.X

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
