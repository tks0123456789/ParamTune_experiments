import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from lasagne.init import GlorotUniform
from nolearn.lasagne import BatchIterator, NeuralNet
from nolearn.lasagne import TrainSplit


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

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def build_net(lr=.02, bs=256, mm=.9, h1=64, h2=128, p1=.2, p2=.2, w_init=0.01,
              max_epochs=10, num_in=37, eval_size=0.0, verbose=0):
              
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
        train_split=TrainSplit(eval_size=eval_size),
        use_label_encoder=True,
        on_epoch_finished = [
            EarlyStopping(patience=10)
        ],
        verbose=verbose,
    )
    return net0

# float64 to float32
def f64_to_f32(params):
    return [np.float32(p) for p in params]

def f64_to_int(params):
    return [int(p) for p in params]
    
# Find good hyper parameters by BayesianOptimization begin
def make_eval_NN(X_train, y_train, X_valid, y_valid, n_folds=5,
                 h1=128, h2=128, bs=256, max_epochs=10,
                 seed=123, verbose=0):
    epoch_lst = []
    score_valid = []
    skf = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True, random_state=seed)
    num_in = X_tr.shape[1]
    def eval_fn(params):
        lr = params['lr']
        mm = params['mm']
        w_init = params['w_init']

        
        score = 0
        epoch = 0
        for tr, va in skf:
            net0 = build_net(lr=lr, bs=bs, mm=mm, h1=h1, h2=h2, p1=p1, p2=p2, w_init=w_init,
                             max_epochs=max_epochs, num_in=num_in, verbose=verbose)
            X_tr, y_tr = X_train[tr], y_train[tr]
            X_va, y_va = X_train[va], y_train[va]
            model.set_params(**params)
            model.fit(X_tr, y_tr)
            hist_last = net0.train_history_[-1]
            score += hist_last['valid_loss']
            epoch += hist_last['epoch']
        score /= n_folds
        epoch /= n_folds
        epoch_lst.append(epoch)
        result_str = "train:%.4f epoch:%5d  " % (score, epoch)
        if X_valid is not None:
            net1 = build_net(lr=lr, bs=bs, mm=mm, h1=h1, h2=h2, p1=p1, p2=p2, w_init=w_init,
                             max_epochs=max_epochs, num_in=num_in, verbose=verbose)
            net1.fit(X_train, y_train)
            pr = net1.predict_proba(X_valid)[:, 1]
            score_valid.append(log_loss(y_valid, pr))
        if verbose:
            print result_str
        return score
    return eval_fn

np.random.seed(321)
SGDC_BO = BayesianOptimization(SGDCcv, {'log_alpha': (-9, 2)}, verbose=0)
gp_params = {'corr': 'absolute_exponential', 'nugget': 1e-7}
SGDC_BO.maximize(n_iter=50, acq='ei', **gp_params)
