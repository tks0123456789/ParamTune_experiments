#!/usr/bin/python
# -*- coding: utf-8 -*-

# 2016/3/2
# i7 4790k 32G GTX660

import pandas as pd
from datetime import datetime
from sklearn.grid_search import ParameterGrid
from hyperopt import hp
from utility_common import path_log
from utility import param_tune_experiment

param_data = {'n_classes':2, 'weights':[.5]}

parameter_space_xgb = {
    "max_depth" : hp.quniform("max_depth", 1, 20, 1),
    "min_child_weight": hp.qloguniform("min_child_weight",0, 8, 1),
    "colsample_bytree": hp.uniform("colsample_bytree",0.1, 1),
}
                    
param_grid = {'n_samples': [5000, 10000, 20000], 'n_features': [10, 20, 40, 80, 160]}

t000 = datetime.now()
rval_lst = []
print "n_samples  n_features     time"
for params in ParameterGrid(param_grid):
    t0 = datetime.now()
    params['n_informative'] = params['n_features'] / 2
    params['n_redundant'] = params['n_informative'] / 2
    param_data.update(params)
    df = param_tune_experiment(param_data, parameter_space_xgb, max_evals=5, n_folds=10, seed=123)
    df.to_csv(path_log + 'r001_n%d_p%d.csv' % (params['n_samples'], params['n_features']))
    rval = params.copy()
    rval['time'] = (datetime.now() - t0).total_seconds()
    rval_lst.append(rval)
    print "%9d %11d %8.2f" % (rval['n_samples'], rval['n_features'], rval['time'])

r001 = pd.DataFrame(rval_lst)
r001.to_csv(path_log + 'r001.csv')

pd.set_option('display.precision', 1)
print 'Time'
print r001.set_index(['n_samples', 'n_features']).time.unstack()

print 'Done', datetime.now() - t000
