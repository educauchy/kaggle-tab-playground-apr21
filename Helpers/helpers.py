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
}

def gen_submit(config):
    print(config)
    output = short_names[config['model']['impute']['type']] + '_' + \
             short_names[config['model']['anomaly']['type']] + '_' + \
             short_names[config['model']['cluster']['type']] + '_' + \
             short_names[config['model']['model']['type']] + '_' + \
             str(random.randint(1, 10000000))
    return output
