from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN


class ClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type='DBSCAN', **params):
        super().__init__()
        self.methods = {
            'DBSCAN': DBSCAN,
        }
        self.method = self.methods[type](**params)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Clustering begins...')
        self.X = X.copy()
        labels = self.method.fit_predict(self.X)
        self.X['Cluster'] = labels
        print('Clustering ended...')
        print('')
        return self.X
