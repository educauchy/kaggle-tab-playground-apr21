from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


class MetaClassifier(ClassifierMixin):
    def __init__(self, model='RF', **params):
        super().__init__()
        self.models = {
            'RF': RandomForestClassifier,
            'LogReg': LogisticRegression,
            'SVM': LinearSVC,
            'AdaBoost': AdaBoostClassifier,
            'GBM': GradientBoostingClassifier,
            'Tree': DecisionTreeClassifier,
        }
        self.model = self.models[model](**params)
        print('Model:')
        print(self.model)
        print('-----------------------')

    def fit(self, X, y=None):
        self.X = X.copy()
        self.y = y.copy()
        self.y.reset_index(drop=True, inplace=True)
        if 'Is_Anomaly' in self.X.columns:
            non_nan_anomaly = self.X[self.X.Is_Anomaly == 1].index
            self.X = self.X[self.X.Is_Anomaly == 1]
            self.y = self.y.reindex(non_nan_anomaly)
        self.X.drop(['Is_Anomaly'], axis=1, inplace=True)
        print(self.X)
        print(self.y)
        self.model.fit(self.X, self.y)
        print('Anomalies found and removed: ' + str(len(non_nan_anomaly)))
        return self

    def predict(self, X, y=None):
        self.X = X.copy()
        self.X.drop(['Is_Anomaly'], axis=1, inplace=True)
        predict = self.model.predict(self.X)
        return predict

