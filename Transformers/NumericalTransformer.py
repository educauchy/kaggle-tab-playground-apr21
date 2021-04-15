# from sklearn.base import BaseEstimator, TransformerMixin


# class NumericalTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         super().__init__()
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         # Numerical features to pass down the numerical pipeline
#         X = X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
#         X = X.replace([np.inf, -np.inf], np.nan)
#         return X.values
