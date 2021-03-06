import random


short_names = {
    'isoforest': 'if',
    'lof': 'lof',
    'onesvm': 'os',
    'iterative': 'iter',
    'KNN': 'knn',
    'simple': 'smpl',
    'DBSCAN': 'dbscn',
    'AdaBoost': 'ada',
    'GBM': 'GBM',
    'RF': 'RF',
    'LGBM': 'LGBM',
}

def gen_submit(config, score_test):
    output = str(score_test) + '_' + \
             short_names[config['model']['impute']['type']] + '_' + \
             short_names[config['model']['anomaly']['type']] + '_' + \
             short_names[config['model']['cluster']['type']] + '_' + \
             short_names[config['model']['model']['type']] + '_' + \
             str(random.randint(1, 10000000))
    return output
