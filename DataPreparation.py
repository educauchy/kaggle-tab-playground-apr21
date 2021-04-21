from Transformers.FeatureSelectionTransformer import FeatureSelectionTransformer
from Transformers.FeatureExtractionTransformer import FeatureExtractionTransformer
from Transformers.ImputeTransformer import ImputeTransformer
from Transformers.EncoderTransformer import EncoderTransformer
from Transformers.AnomalyDetectionTransformer import AnomalyDetectionTransformer
from Transformers.DropColumnsTransformer import DropColumnsTransformer
from Transformers.ClusteringTransformer import ClusteringTransformer
from Models.MetaClassifier import MetaClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
# import entropy_based_binning as ebb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer, make_column_selector



train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train_X = train[train.columns[~train.columns.isin(['Survived'])]]
train_y = train['Survived']

X = train[train.columns[~train.columns.isin(['Survived'])]]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)

out_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Family_Members', 'Fare', \
        'Gender', 'Surname_Enc', 'Origin', 'Is_Alone', 'Fare_Log', 'Age_Bins']

test['Survived'] = 0
full = pd.concat([train, test])
all_surnames = full['Name'].str.split(", ", expand=True)[0]
all_names = full['Name'].str.split(", ", expand=True)[1]


preprocess_pipeline = Pipeline(steps=[
    ('sex_encoder', EncoderTransformer(type='label', column='Sex', out_column='Gender')),
    ('embarked_encoder', EncoderTransformer(type='label', column='Embarked', out_column='Origin')),
    ('cabin_letter_encoder', EncoderTransformer(type='label', column='Cabin_Letter', out_column='Cabin_Letter_Enc')),
    ('surname_encoder', EncoderTransformer(type='label', column='Surname', out_column='Surname_Enc', data=all_surnames)),
    ('firstname_encoder', EncoderTransformer(type='label', column='Firstname', out_column='Firstname_Enc', data=all_names)),
])

estimators = [
    ('bg_knn', BaggingClassifier(KNeighborsClassifier(), n_estimators=100, bootstrap_features=True, oob_score=True, \
                                n_jobs=-1, random_state=0)),
    ('bg_logreg', BaggingClassifier(LogisticRegression(max_iter=10000), n_estimators=100, bootstrap_features=True, \
                                    oob_score=True, n_jobs=-1, random_state=0)),
    ('rf_dt', RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=0)),
    ('ada_dt', AdaBoostClassifier(n_estimators=100, random_state=0)),
]
full_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumnsTransformer(columns=['Ticket', 'PassengerId'])),
    ('feature_extraction', FeatureExtractionTransformer(age_bins=range(0, 100, 10))),
    ('preprocess_pipeline', preprocess_pipeline),
    ('feature_selection', FeatureSelectionTransformer(columns=out_columns)),
    # ('imputation', ImputeTransformer(type='simple', strategy='median')),
    ('imputation', ImputeTransformer(type='KNN', n_neighbors=5)), # 2-76.33; 3-76.3; 4-76.33
    # ('imputation', ImputeTransformer(type='iterative', max_iter=25, sample_posterior=True)),
    ('anomaly_detection', AnomalyDetectionTransformer(type='isoforest', columns=out_columns, n_estimators=100)),
    # ('anomaly_detection', AnomalyDetectionTransformer(type='lof', columns=out_columns, n_neighbors=20)),
    # ('anomaly_detection', AnomalyDetectionTransformer(type='onesvm', columns=out_columns)),
    ('clustering', ClusteringTransformer(type='DBSCAN', eps=1.1, n_jobs=-1)),
    ('model', MetaClassifier(model='RF', n_estimators=1000, criterion='entropy', \
                                random_state=0, oob_score=True, n_jobs=-1)),
    # ('model', MetaClassifier(model='GBM', n_estimators=500)),
    # ('model', MetaClassifier(model='AdaBoost', n_estimators=2500)),
    # ('model', LGBMClassifier(n_estimators=2000, max_depth=1, n_jobs=-1)),
    # ('model', MetaClassifier(model='Stacking', estimators=estimators, cv=4, n_jobs=-1, \
    #                          final_estimator=LinearSVC(max_iter=50000)))
])


cv = KFold(n_splits=4, shuffle=True, random_state=0)
scores = cross_val_score(full_pipeline, train_X, train_y, cv = cv)
print(scores)

# training = full_pipeline.fit(X_train, y_train)
# score_test = round(training.score(X_test, y_test) * 100, 2)
# print('----------------------------')
# print('Score: ' + str(score_test))

# out = pd.DataFrame(data={'PassengerId': test['PassengerId'].astype(int)})
# out['Survived'] = training.predict(test).astype(int)
# out.to_csv('./submissions/Pipeline_LGBM_Full_' + str(random.random()) + '.csv', index=False)


# print(pd.DataFrame(np.array(list(zip(cols, np.round(clf.feature_importances_, 3)))), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False))



param_grid = {
    # 'imputation__n_neighbors': range(1, 10),
    # 'anomaly_detection__n_estimators': range(100, 1000, 100),
    # 'model__n_estimators': range(100, 2000, 100),
    'clustering__eps': range(1, 10),
}
# grid = GridSearchCV(full_pipeline, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
# grid.fit(X_train, y_train)
# print(grid)
# print(grid.best_params_)
# print(grid.score(X_test, y_test))

