from Transformers.FeatureSelectionTransformer import FeatureSelectionTransformer
from Transformers.FeatureExtractionTransformer import FeatureExtractionTransformer
from Transformers.ImputeTransformer import ImputeTransformer
from Transformers.EncoderTransformer import EncoderTransformer
from Transformers.AnomalyDetectionTransformer import AnomalyDetectionTransformer
from Transformers.DropColumnsTransformer import DropColumnsTransformer
from Transformers.ClusteringTransformer import ClusteringTransformer
from Transformers.FeatureInteractionTransformer import FeatureInteractionTransformer
from Models.MetaClassifier import MetaClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
import pandas as pd
import yaml
import os, sys
from shutil import copyfile
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
all = {
    'Surname': full['Name'].str.split(", ", expand=True)[0],
    'Firstname': full['Name'].str.split(", ", expand=True)[1],
}


preprocess_steps = []
for item in config['model']['encoding']:
    data = all[item['column']] if item['data'] else None
    encoder = EncoderTransformer(type=item['type'], column=item['column'], out_column=item['out_column'], data=data)
    preprocess_steps.append( (item['column'] + '_encoder', encoder) )
preprocess_pipeline = Pipeline(steps=preprocess_steps)

stacking_estimators = [
    ('rf', RandomForestClassifier(n_estimators=2000, max_depth=30, random_state=config['model']['random_state'])),
    ('bg_lr', BaggingClassifier(base_estimator=LogisticRegression(max_iter=10000), n_estimators=2000, bootstrap_features=True, random_state=config['model']['random_state'])),
    ('bg_svm', BaggingClassifier(base_estimator=LinearSVC(max_iter=10000), n_estimators=2000, bootstrap_features=True, random_state=config['model']['random_state'])),
]


full_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumnsTransformer(columns=config['model']['drop_columns'])),
    ('f_extraction', FeatureExtractionTransformer(age_bins=config['model']['f_ext']['age_bins'])),
    ('preprocess', preprocess_pipeline),
    ('f_selection', FeatureSelectionTransformer(columns=config['model']['out_columns'])),
    ('impute', ImputeTransformer(type=config['model']['impute']['type'], \
                                     **config['model']['impute']['params'])),
    ('f_inter', FeatureInteractionTransformer(**config['model']['f_inter']['params'])),
    ('anomaly', AnomalyDetectionTransformer(type=config['model']['anomaly']['type'], \
                                            columns=config['model']['out_columns'], \
                                            **config['model']['anomaly']['params'])),
    ('cluster', ClusteringTransformer(type=config['model']['cluster']['type'], \
                                    **config['model']['cluster']['params'])),
    # ('model', MetaClassifier(model=config['model']['model']['type'], \
    #                          random_state=config['model']['random_state'], \
    #                          **config['model']['model']['params'])),
    # OR
    ('model', StackingClassifier(estimators=stacking_estimators, cv=10, final_estimator=LogisticRegression(max_iter=10000), n_jobs=-1)),
])



if config['model']['strategy'] == 'cv':
    cv = KFold(n_splits=config['model']['KFold_folds'], shuffle=True, random_state=config['model']['random_state'])
    scores = cross_val_score(full_pipeline, X, y, cv = cv)
    print('KFold scores:')
    print(scores)
elif config['model']['strategy'] == 'grid_search':
    training = GridSearchCV(full_pipeline, config['model']['param_grid'], scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    training.fit(X, y)
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






