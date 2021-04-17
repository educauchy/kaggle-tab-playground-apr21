from Transformers.FeatureSelectionTransformer import FeatureSelectionTransformer
from Transformers.FeatureExtractionTransformer import FeatureExtractionTransformer
from Transformers.ImputeTransformer import ImputeTransformer
from Transformers.EncoderTransformer import EncoderTransformer
from Transformers.AnomalyDetectionTransformer import AnomalyDetectionTransformer
from Transformers.DropColumnsTransformer import DropColumnsTransformer
from Models.MetaClassifier import MetaClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, FeatureUnion
# from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
# import xgboost as xgb
# import lightgbm as lgb



train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

X = train[train.columns[~train.columns.isin(['Survived'])]]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

out_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Family_Members', 'Fare', \
        'Gender', 'Surname_Enc', 'Origin', 'Is_Alone', 'Fare_Log', 'Age_Bins', 'Cabin_Letter_Enc']

test['Survived'] = 0
full = pd.concat([train, test])
all_surnames = full['Name'].str.split(", ", expand=True)[0]



preprocess_pipeline = Pipeline(steps=[
    ('sex_encoder', EncoderTransformer(type='label', column='Sex', out_column='Gender')),
    ('embarked_encoder', EncoderTransformer(type='label', column='Embarked', out_column='Origin')),
    ('cabin_letter_encoder', EncoderTransformer(type='label', column='Cabin_Letter', out_column='Cabin_Letter_Enc')),
    ('surname_encoder', EncoderTransformer(type='label', column='Surname', out_column='Surname_Enc', data=all_surnames)),
    # ('full_pipeline', union_pipeline),
    # ('cat_pipe', categorical_transformers),
# )
])


full_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumnsTransformer(columns=['Ticket'])),
    ('feature_extraction', FeatureExtractionTransformer()),
    ('preprocess_pipeline', preprocess_pipeline),
    ('feature_selection', FeatureSelectionTransformer(columns=out_columns)),
    ('imputation', ImputeTransformer(type='KNN', n_neighbors=5)),
    # ('imputation', ImputeTransformer(type='iterative', initial_strategy='median')),
    # ('anomaly_detection', AnomalyDetectionTransformer(type='isoforest', columns=out_columns, n_estimators=300 )),
    ('anomaly_detection', AnomalyDetectionTransformer(type='lof', columns=out_columns, n_neighbors=5, novelty=True)),
    # ('anomaly_detection', AnomalyDetectionTransformer(type='onesvm', columns=out_columns)),
    # ('model', MetaClassifier(model='RF', n_estimators=1000, criterion='entropy', \
    #                             random_state=0, oob_score=True, n_jobs=-1)), # 74.39
    ('model', MetaClassifier(model='GBM', n_estimators=500)), # 72.36 -> 76.85
    # ('grid_search', GridSearchCV(param_grid=param_grid, cv=5, n_jobs=-1, verbose=4)),
])

training = full_pipeline.fit(X_train, y_train)
score_test = round(training.score(X_test, y_test) * 100, 2)
print('----------------------------')
print('Score: ' + str(score_test))


# param_grid = {
#     'anomaly_detection__n_estimators': range(1, 4, 2),
#     'model__n_estimators': range(1, 4, 2),
# }
# grid = GridSearchCV(full_pipeline, param_grid, cv=5, verbose=3, n_jobs=-1)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.score(X_test, y_test))




# numerical_pipeline = Pipeline(steps=[
# ('num_selector', make_column_selector(dtype_include=[float, int])),
# ('num_transformer', NumericalTransformer()),
# ('imputer', KNNImputer(n_neighbors=5)),
# ('std_scaler', StandardScaler())
# ])


# categorical_transformers = ColumnTransformer(transformers=[
# ('Surname_Enc', LabelEncoder(), ['Surname']),
# ('Cabin_Letter_Enc', LabelEncoder(), ['Cabin_Letter']),
# ('cat_selector', make_column_selector(dtype_include="category")),
# ('Age_Bins', FunctionTransformer(age_to_bins), ['Age']),
# ('Fare_Log', FunctionTransformer(np.log1p), ['Fare']),
# ])


# union_pipeline = FeatureUnion(transformer_list=[
#     ('categorical_pipeline', categorical_transformers),
#     ('numerical_pipeline', numerical_pipeline)
# ])

