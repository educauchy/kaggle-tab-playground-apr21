from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class AnomalyDetectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type='isoforest', n=300):
        super().__init__()
        self.detectors = {
            'isoforest': IsolationForest(n_estimators=n, n_jobs=-1),
        }
        self.detector = self.detectors[type]

    def fit(self, X, y=None):
        self.detector.fit(X)
        print(y)
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        pred_anomaly = self.detector.predict(self.X)
        self.X = self.X[pred_anomaly == 1]
        self.X.reset_index(drop=True, inplace=True)
        print(y)
        return self.X
