from Transformers import NumericalTransformer, CategoricalTransformer, FeatureExtraction
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import pandas as pd


data = pd.read_csv('./data/train.csv')

X = data[data.columns[~data.columns.isin(['Survived'])]]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


numerical_pipeline = Pipeline(steps=[
    ('num_transformer', NumericalTransformer()),
    ('imputer', KNNImputer(n_neighbors=5)),
    ('std_scaler', StandardScaler())])

categorical_pipeline = Pipeline(steps=[
    ('cat_transformer', CategoricalTransformer()),
    ('imputer', KNNImputer(n_neighbors=5)),
    ('one_hot_encoder', OneHotEncoder(sparse=False))])

union_pipeline = FeatureUnion(transformer_list=[
    ('categorical_pipeline', categorical_pipeline),
    ('numerical_pipeline', numerical_pipeline)])

preprocess_pipeline = Pipeline(steps=[('anomaly_detection', IsolationForest(n_estimators=500)),
                                      ('full_pipeline', union_pipeline)])

# define full pipeline --> preprocessing + model
full_pipeline = Pipeline(steps=[
    ('feature_extraction', FeatureExtraction())
    ('preprocess_pipeline', preprocess_pipeline),
    ('model', DecisionTreeClassifier())])

# fit on the complete pipeline
training = full_pipeline.fit(X_train, y_train)
print(full_pipeline.get_params())

# metrics
score_test = \
    round(training.score(X_test, y_test) * 100, 2)
print(f"\nTraining Accuracy: {score_test}")