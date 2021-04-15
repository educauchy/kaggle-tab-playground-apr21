from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.out_columns = ['Pclass', 'Family_Members', 'Gender', 'Origin', \
                            'Surname_Enc', 'Cabin_Letter_Enc', \
                            'Age', 'Fare', 'Fare_Log', 'Is_Alone', 'Age_Bins']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.out_columns]
