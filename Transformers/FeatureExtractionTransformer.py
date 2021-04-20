from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.stats import boxcox


class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, age_bins=()):
        super().__init__()
        self.age_bins = age_bins

    def __age_binning(self, col, bins=()):
        labels = range(len(bins) - 1)
        age_bins = pd.cut(col, bins=bins, labels=labels)
        return age_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X['Family_Members'] = self.X['SibSp'] + self.X['Parch']
        self.X['Is_Alone'] = np.where(self.X['Family_Members'] == 0, 1, 0)
        self.X[['Surname', 'Firstname']] = self.X['Name'].str.split(", ", expand=True)
        self.X['Cabin_Letter'] = X['Cabin'].str.slice(0, 1)
        self.X['Age_Bins'] = self.__age_binning(self.X['Age'], bins=self.age_bins)
        # age_log = np.log1p(self.X['Age'])
        # self.X['Age_Log'] = age_log[age_log > 2.7]
        self.X['Fare_Log'] = np.log1p(X['Fare'])
        return self.X
