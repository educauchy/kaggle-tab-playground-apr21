from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
import numpy as np


class AnomalyDetectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type='isoforest', columns=(), n=300):
        super().__init__()
        self.detectors = {
            'isoforest': IsolationForest(n_estimators=n, n_jobs=-1),
        }
        self.columns = columns
        self.detector = self.detectors[type]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        pred_anomaly = self.detector.fit_predict(X)
        self.X['Is_Anomaly'] = pred_anomaly
        return self.X
