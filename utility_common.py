import pandas as pd

path_log = 'log/'

def summary(trials, n_estimators_lst, score_valid):
    df = pd.DataFrame(trials.vals)
    df['train'] = trials.losses()
    df['n_estimators'] = n_estimators_lst
    df['valid'] = score_valid
    return df


