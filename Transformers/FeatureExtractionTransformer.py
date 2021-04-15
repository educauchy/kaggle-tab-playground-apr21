from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X['Family_Members'] = self.X['SibSp'] + self.X['Parch']
        self.X['Is_Alone'] = np.where(self.X['Family_Members'] == 0, 1, 0)
        self.X['Surname'] = X['Name'].str.split(", ", expand=True)[0]
        self.X['Cabin_Letter'] = X['Cabin'].str.slice(0, 1)
        self.X['Age_Bins'] = np.where(X['Age'] <= 18, 1,
                            np.where(X['Age'] <= 30, 2,
                                     np.where(X['Age'] <= 45, 3,
                                              np.where(X['Age'] <= 60, 4,
                                                       np.where(X['Age'] <= 80, 5, 6)))))
        self.X['Fare_Log'] = np.log1p(X['Fare'])
        return self.X
