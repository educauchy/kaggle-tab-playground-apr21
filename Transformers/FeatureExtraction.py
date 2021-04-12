from sklearn.base import BaseEstimator, TransformerMixin


# data['Family_Members'] = data['SibSp'] + data['Parch']
# data[['Surname', 'Firstname']] = data.Name.str.split(", ", expand=True)
# data['Cabin_Letter'] = data.Cabin.str.slice(0, 1)
# data['Cabin_Number'] = data.Cabin.str.slice(1)
# data_train['Family_Members'] = data_train['SibSp'] + data_train['Parch']
# data_train['Fare_Log'] = np.log2(data_train['Fare'])
# data_train['Is_Alone'] = np.where(data_train['Family_Members'] == 0, 1, 0)
# data_train['Age_Bins'] = np.where(data_train['Age'] <= 18, 1,
#                                  np.where(data_train['Age'] <= 30, 2,
#                                  np.where(data_train['Age'] <= 45, 3,
#                                  np.where(data_train['Age'] <= 60, 4,
#                                  np.where(data_train['Age'] <= 80, 5, 6)))))
# data_train['Age'] = StandardScaler().fit_transform(data_train[['Age']])
# data_train['Fare'] = StandardScaler().fit_transform(data_train[['Fare']])


class FeatureExtraction(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.means_ = None
        self.std_ = None
        self.X = None

    def fit(self, X, y=None):
        X = X.to_numpy()
        self.means_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)

        return self

    def transform(self, X, y=None):
        self.X[:] = (X.to_numpy() - self.means_) / self.std_

        return self.X