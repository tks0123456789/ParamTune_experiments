import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator, NeuralNet
from nolearn.lasagne import TrainSplit

from bayes_opt import BayesianOptimization

class BatchIterator_shuffle(object):
    def __init__(self, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        bs = self.batch_size
        idx = range(self.n_samples)
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range((self.n_samples + bs - 1) // bs):
            n_last = min((i + 1) * bs, self.n_samples)
            sl = slice(i * bs, n_last)
            Xb = self.X[idx[sl]]
            if self.y is not None:
                yb = self.y[idx[sl]]
            else:
                yb = None
            yield self.transform(Xb, yb)

    @property
    def n_samples(self):
        X = self.X
        return X.shape[0]

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state

def build_net(lr=.02, bs=256, mm=.9, h1=64, h2=128, p1=.2, p2=.2, w_init=0.01,
              max_epochs=10, num_in=37, verbose=0):
              
    layers0 = [
        (InputLayer, {'shape': (None, num_in)}),
        (DenseLayer, {'num_units': h1,
                      'W':GlorotUniform(w_init)}),
        (DropoutLayer, {'p': p1}),
        (DenseLayer, {'num_units': h2,
                      'W':GlorotUniform(w_init)}),
        (DropoutLayer, {'p': p2}),
        (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]

    net0 = NeuralNet(
        layers=layers0,
        batch_iterator_train=BatchIterator_shuffle(batch_size=bs, shuffle=True),
        batch_iterator_test=BatchIterator(batch_size=bs),
        max_epochs=max_epochs,
        update=nesterov_momentum,
        update_momentum=mm,
        update_learning_rate=lr,
        regression=False,
        train_split=TrainSplit(eval_size=0.0),
        use_label_encoder=True,
        verbose=verbose,
    )
    return net0

# float64 to float32
def f64_to_f32(params):
    return [np.float32(p) for p in params]

def f64_to_int(params):
    return [int(p) for p in params]
    
# Find good hyper parameters by BayesianOptimization begin
def make_eval_NN(X_tr, y_tr, X_va, y_va, h1=128, h2=128, bs=256, max_epochs=10, n_iter_predict=5,
                 seed_base=123, verbose=0):
    num_in = X_tr.shape[1]
    def eval_NN(log_lr, log_w_init, mm, p1, p2):
        lr, mm, p1, p2, w_init = f64_to_f32((np.exp(log_lr),mm, p1, p2, np.exp(log_w_init)))
                                             
        #h1, h2 = f64_to_int((h1, h2))
        pr = np.zeros(X_va.shape[0])
        for seed in range(seed_base, seed_base+n_iter_predict):
            np.random.seed(seed)
            net0 = build_net(lr=lr, bs=bs, mm=mm, h1=h1, h2=h2, p1=p1, p2=p2, w_init=w_init,
                             max_epochs=max_epochs, num_in=num_in, verbose=verbose)
            net0.fit(X_tr, y_tr)
            pr1 = net0.predict_proba(X_va)[:, 1]
            pr += pr1
            print roc_auc_score(y_va, pr)
        return roc_auc_score(y_va, pr)
    return eval_NN

np.random.seed(321)
SGDC_BO = BayesianOptimization(SGDCcv, {'log_alpha': (-9, 2)}, verbose=0)
gp_params = {'corr': 'absolute_exponential', 'nugget': 1e-7}
SGDC_BO.maximize(n_iter=50, acq='ei', **gp_params)
