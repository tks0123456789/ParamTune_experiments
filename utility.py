from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.datasets import make_classification
from xgboost.sklearn import XGBClassifier
from hyperopt import fmin, tpe, hp, rand, Trials
from utility_common import summary

# Usage:
# parameter_space_xgb = {
#     "max_depth" : hp.quniform("max_depth", 1, 12, 1),
#     "min_child_weight": hp.qloguniform("min_child_weight",0, 7, 1),
#     "colsample_bytree": hp.uniform("colsample_bytree",0.1, 1),
# }
# trials = Trials()
# eval_fn = make_eval_XGBC_hyperopt(X_train, y_train, X_valid, y_valid,
#                          subsample=.8, n_folds=5, seed=321)
# best = fmin(eval_fn, parameter_space_xgb, algo=tpe.suggest, trials=trials, max_evals=200)

# For hyperopt
def make_eval_XGBC(X_train, y_train, X_valid, y_valid,
                   n_estimators_max=500, learning_rate=0.1,
                   n_folds=5, subsample=.8, seed=123, verbose=False):
    n_estimators_lst = []
    score_valid = []
    skf = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True, random_state=seed)
    def eval_fn(params):
        model = XGBClassifier(n_estimators=n_estimators_max, learning_rate=learning_rate, seed=seed)
        score = 0
        n_estimators = 0
        for tr, va in skf:
            X_tr, y_tr = X_train[tr], y_train[tr]
            X_va, y_va = X_train[va], y_train[va]
            model.set_params(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='logloss',
                      early_stopping_rounds=50, verbose=False)
            score += model.best_score
            n_estimators += model.best_iteration
        score /= n_folds
        n_estimators /= n_folds
        n_estimators_lst.append(n_estimators)
        result_str = "train:%.4f ntree:%5d  " % (score, n_estimators)
        if X_valid is not None:
            model.n_estimators = n_estimators
            model.fit(X_train, y_train)
            pr = model.predict_proba(X_valid)[:,1]
            sc_valid = log_loss(y_valid, pr)
            score_valid.append(sc_valid)
            result_str += "valid:%.4f" % sc_valid
        if verbose:
            print result_str
        return score
    return eval_fn, n_estimators_lst, score_valid

def param_tune_experiment(param_data, param_space_xgb, max_evals=10, n_folds=5,
                          seed=123, verbose=False):
    X, y = make_classification(**param_data)
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=.2, random_state=seed)
    train_idx, valid_idx = list(sss)[0]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    
    eval_fn, n_lst, sc_valid= make_eval_XGBC(X_train, y_train, X_valid=X_valid, y_valid=y_valid,
                                             n_estimators_max=5000, learning_rate=.1,
                                             subsample=.8, n_folds=n_folds, seed=321, verbose=verbose)
    trials = Trials()
    best = fmin(eval_fn, param_space_xgb, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    if verbose:
        print best, trials.best_trial['result']['loss']
    df = summary(trials, n_lst, sc_valid)
    return df
