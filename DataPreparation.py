from Transformers import NumericalTransformer, CategoricalTransformer
from Transformers.FeatureSelectionTransformer import FeatureSelectionTransformer
from Transformers.FeatureExtractionTransformer import FeatureExtractionTransformer
from Transformers.ImputeTransformer import ImputeTransformer
from Transformers.EncoderTransformer import EncoderTransformer
from Transformers.AnomalyDetectionTransformer import AnomalyDetectionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, FeatureUnion
# from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, FunctionTransformer, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import pandas as pd
import numpy as np



train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

X = train[train.columns[~train.columns.isin(['Survived'])]]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

test['Survived'] = 0
full = pd.concat([train, test])
all_surnames = full['Name'].str.split(", ", expand=True)[0]

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

preprocess_pipeline = Pipeline(steps=[
                                # ('full_pipeline', union_pipeline),
                                # ('cat_pipe', categorical_transformers),
                                ('sex_encoder', EncoderTransformer(type='label', column='Sex', \
                                                        out_column='Gender')),
                                ('embarked_encoder', EncoderTransformer(type='label', column='Embarked', \
                                                        out_column='Origin')),
                                ('cabin_letter_encoder', EncoderTransformer(type='label', column='Cabin_Letter', \
                                                        out_column='Cabin_Letter_Enc')),
                                ('surname_encoder', EncoderTransformer(type='label', column='Surname', \
                                                        out_column='Surname_Enc', data=all_surnames)),
                                # ('anomaly_detection', IsolationForest(n_estimators=500))
])


full_pipeline = Pipeline(steps=[
    ('feature_extraction', FeatureExtractionTransformer()),
    ('preprocess_pipeline', preprocess_pipeline),
    ('feature_selection', FeatureSelectionTransformer()),
    # ('imputation', ImputeTransformer(type='KNN', n_neighbors=5)),
    ('imputation', ImputeTransformer(type='iterative')),
    ('anomaly_detection', AnomalyDetectionTransformer(type='isoforest', n=50)),
    ('model', RandomForestClassifier(n_estimators=5, n_jobs=-1)),
])

training = full_pipeline.fit(X_train, y_train)
# print(full_pipeline.get_params())

# training_model = training.steps[4][1]
# print(training_model.feature_importances_)
#
# print(training_model)
# print(training.feature_importances_)

# score_test = \
#     round(training.score(X_test, y_test) * 100, 2)
# print(f"\nTraining Accuracy: {score_test}")