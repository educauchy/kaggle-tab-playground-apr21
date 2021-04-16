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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1488)

out_columns = ['Pclass', 'Family_Members', 'Gender', 'Origin', 'Surname_Enc', \
                    'Cabin_Letter_Enc', \
                    'Age', 'Fare', 'Fare_Log', 'Is_Alone', 'Age_Bins']

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
    ('imputation', ImputeTransformer(type='KNN', n_neighbors=8)),
    ('anomaly_detection', AnomalyDetectionTransformer(type='isoforest', columns=out_columns, n=200)),
    # ('model', MetaClassifier(model='RF', n_estimators=2)),
    ('model', MetaClassifier(model='GBM', n_estimators=200)),
])



training = full_pipeline.fit(X_train, y_train)
score_test = \
    round(training.score(X_test, y_test) * 100, 2)
print(f"\nTraining Accuracy: {score_test}")
# print(full_pipeline.get_params())

# training_model = training.steps[5]
# print(training_model)
#
# print(training.feature_importances_)

# pg = [
#     {
#         'classify': [LinearSVC()],
#         'classify__penalty': ['l1', 'l2']
#     },
#     {
#         'classify': [DecisionTreeClassifier()],
#         'classify__min_samples_split': [2, 10, 20]
#     },
# ]
# grid_search = GridSearchCV(full_pipeline, param_grid=pg, cv=3)
# print(grid_search.score(X_test, y_test))

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

