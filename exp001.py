#!/usr/bin/python
# -*- coding: utf-8 -*-

# 2016/3/2
# i7 4790k 32G GTX660

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.grid_search import ParameterGrid
from hyperopt import hp
from utility_common import path_log
from utility import param_tune_experiment

t000 = datetime.now()

param_data = {'n_classes':2, 'weights':[.5]}

parameter_space_xgb = {
    "max_depth" : hp.quniform("max_depth", 1, 20, 1),
    "min_child_weight": hp.qloguniform("min_child_weight",0, 8, 1),
    "colsample_bytree": hp.uniform("colsample_bytree",0.1, 1),
}
                    
param_grid = {'n_samples': [5000, 10000, 20000], 'n_features': [10, 20, 40, 80]}

rval_lst = []
print "n_samples  n_features  train  valid     time"
for params in ParameterGrid(param_grid):
    t0 = datetime.now()
    params['n_informative'] = params['n_features'] / 2
    params['n_redundant'] = params['n_informative'] / 2
    param_data.update(params)
    np.random.seed(3210)
    df = param_tune_experiment(param_data, parameter_space_xgb, max_evals=100, n_folds=5, seed=123)
    df.to_csv(path_log + 'r001_n%d_p%d.csv' % (params['n_samples'], params['n_features']))
    best = df.ix[df.train.idxmin()].to_dict()
    rval = params.copy()
    rval.update(best)
    rval['time'] = (datetime.now() - t0).total_seconds()
    rval_lst.append(rval)
    print "%9d %11d %.4f %.4f %8.2f" % \
        (rval['n_samples'], rval['n_features'], rval['train'], rval['valid'], rval['time'])

r001 = pd.DataFrame(rval_lst)
r001.to_csv(path_log + 'r001.csv')

result = r001.set_index(['n_samples', 'n_features'])

pd.set_option('display.precision', 1)
print '\nTime'
print result.time.unstack()

print '\nDone', datetime.now() - t000
