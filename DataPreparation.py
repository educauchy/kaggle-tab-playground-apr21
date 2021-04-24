from Transformers.FeatureSelectionTransformer import FeatureSelectionTransformer
from Transformers.FeatureExtractionTransformer import FeatureExtractionTransformer
from Transformers.ImputeTransformer import ImputeTransformer
from Transformers.EncoderTransformer import EncoderTransformer
from Transformers.AnomalyDetectionTransformer import AnomalyDetectionTransformer
from Transformers.DropColumnsTransformer import DropColumnsTransformer
from Transformers.ClusteringTransformer import ClusteringTransformer
from Transformers.FeatureInteractionTransformer import FeatureInteractionTransformer
from Models.MetaClassifier import MetaClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import yaml
import os, sys
from shutil import copyfile
import random
from sklearn.model_selection import KFold, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer, make_column_selector
from Helpers.helpers import gen_submit


try:
    project_dir = os.path.dirname(__file__)
    config_file = os.path.join(project_dir, 'config/config.yaml')

    with open (config_file, 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    print(exc)
    sys.exit(1)
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)



train = pd.read_csv(config['data']['train'])
test = pd.read_csv(config['data']['test'])
X = train[train.columns[~train.columns.isin([config['data']['target']])]]
y = train[config['data']['target']]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config['data']['train_size'], \
                                                    random_state=config['model']['random_state'])

# Join train and test data for encoding
test[config['data']['target']] = 0
full = pd.concat([train, test])
all_surnames = full['Name'].str.split(", ", expand=True)[0]
all_names = full['Name'].str.split(", ", expand=True)[1]



preprocess_pipeline = Pipeline(steps=[
    ('sex_encoder', EncoderTransformer(type='label', column='Sex', out_column='Gender')),
    ('embarked_encoder', EncoderTransformer(type='label', column='Embarked', out_column='Origin')),
    ('cabin_letter_encoder', EncoderTransformer(type='label', column='Cabin_Letter', out_column='Cabin_Letter_Enc')),
    ('surname_encoder', EncoderTransformer(type='label', column='Surname', out_column='Surname_Enc', data=all_surnames)),
    ('firstname_encoder', EncoderTransformer(type='label', column='Firstname', out_column='Firstname_Enc', data=all_names)),
    ('pcl_emb_encoder', EncoderTransformer(type='label', column='Pclass_Embarked', out_column='Pcl_Emb_Enc')),
    ('pcl_sex_encoder', EncoderTransformer(type='label', column='Pclass_Sex', out_column='Pcl_Sex_Enc')),
    ('sex_emb_encoder', EncoderTransformer(type='label', column='Sex_Embarked', out_column='Sex_Emb_Enc')),
])

full_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumnsTransformer(columns=config['model']['drop_columns'])),
    ('f_extraction', FeatureExtractionTransformer(age_bins=config['model']['f_ext']['age_bins'])),
    ('preprocess', preprocess_pipeline),
    ('f_selection', FeatureSelectionTransformer(columns=config['model']['out_columns'])),
    ('impute', ImputeTransformer(type=config['model']['impute']['type'], \
                                     **config['model']['impute']['params'])),
    ('f_interaction', FeatureInteractionTransformer(interaction_only=True, include_bias=False)),
    ('anomaly', AnomalyDetectionTransformer(type=config['model']['anomaly']['type'], \
                                            columns=config['model']['out_columns'], \
                                            **config['model']['anomaly']['params'])),
    ('cluster', ClusteringTransformer(type=config['model']['cluster']['type'], \
                                    **config['model']['cluster']['params'])),
    ('model', MetaClassifier(model=config['model']['model']['type'], \
                             random_state=config['model']['random_state'], \
                             **config['model']['model']['params'])),
])



if config['model']['strategy'] == 'cv':
    cv = KFold(n_splits=config['model']['KFold_folds'], shuffle=True, random_state=config['model']['random_state'])
    scores = cross_val_score(full_pipeline, X, y, cv = cv)
    print('KFold scores:')
    print(scores)
elif config['model']['strategy'] == 'grid_search':
    param_grid = {
        'imputation__n_neighbors': range(1, 10),
        'anomaly_detection__n_estimators': range(100, 1000, 100),
        'model__n_estimators': range(100, 2000, 100),
        'clustering__eps': range(1, 10),
    }
    training = GridSearchCV(full_pipeline, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    training.fit(X_train, y_train)
    print(training)
    print(training.best_params_)
    print('Score: ' + str(training.score(X_test, y_test)))
elif config['model']['strategy'] == 'model':
    training = full_pipeline.fit(X_train, y_train)
    score_test = round(training.score(X_test, y_test) * 100, 2)
    print('Score: ' + str(score_test))


if config['output']['save']:
    out = pd.DataFrame(data={'PassengerId': test['PassengerId'].astype(int)})
    out['Survived'] = training.predict(test).astype(int)

    output_folder = gen_submit(config, score_test)
    output_path = os.path.join(project_dir, 'submissions', output_folder)
    os.mkdir(output_path)

    out.to_csv( os.path.join(output_path, 'output.csv'), index=False)
    copyfile( config_file, os.path.join(output_path, 'config.yaml') )






