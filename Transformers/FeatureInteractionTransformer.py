from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np


class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, not_include=(), degree=2, interaction_only=False, include_bias=True):
        super().__init__()
        self.not_include = not_include
        self.creator = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)

    def fit(self, X, y=None):
        self.creator.fit(X)
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        new_values = np.nan_to_num((self.creator.transform(self.X).astype(np.float32)))
        d = pd.DataFrame(data=new_values, columns=self.creator.get_feature_names())
        self.X = self.X.join(d)
        return self.X
