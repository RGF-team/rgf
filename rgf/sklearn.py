from __future__ import absolute_import

__all__ = ('RGFClassifier', 'RGFRegressor')

from glob import glob
from math import ceil
from threading import Lock
from uuid import uuid4
import atexit
import numbers
import os
import platform
import subprocess

import numpy as np
from scipy.sparse import isspmatrix
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.externals import six
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_X_y, column_or_1d

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()

_ALGORITHMS = ("RGF", "RGF_Opt", "RGF_Sib")
_LOSSES = ("LS", "Expo", "Log")
_FLOATS = (float, np.float, np.float16, np.float32, np.float64, np.double)
_SYSTEM = platform.system()
_UUIDS = []


## Edit this ##################################################
if _SYSTEM in ('Windows', 'Microsoft'):
    # Location of the RGF executable
    loc_exec = 'C:\\Program Files\\RGF\\bin\\rgf.exe'
    # Location for RGF temp files
    loc_temp = 'temp/'
    default_exec = 'rgf.exe'
else:  # Linux, Darwin (OS X), etc.
    # Location of the RGF executable
    loc_exec = '/opt/rgf1.2/bin/rgf'
    # Location for RGF temp files
    loc_temp = '/tmp/rgf'
    default_exec = 'rgf'
## End Edit ##################################################


def _is_executable_response(path):
    try:
        subprocess.check_output((path, "train"))
        return True
    except Exception:
        return False

# Validate path
if _is_executable_response(default_exec):
    loc_exec = default_exec
elif not os.path.isfile(loc_exec):
    raise Exception('{0} is not executable file. Please set '
                    'loc_exec to RGF execution file.'.format(loc_exec))
elif not os.access(loc_exec, os.X_OK):
    raise Exception('{0} cannot be accessed. Please set '
                    'loc_exec to RGF execution file.'.format(loc_exec))
elif _is_executable_response(loc_exec):
    pass
else:
    raise Exception('{0} does not exist or {1} is not in the '
                    '"PATH" variable.'.format(loc_exec, default_exec))

if not os.path.isdir(loc_temp):
    os.makedirs(loc_temp)
if not os.access(loc_temp, os.W_OK):
    raise Exception('{0} is not writable directory. Please set '
                    'loc_temp to writable directory'.format(loc_temp))


@atexit.register
def _cleanup():
    for uuid in _UUIDS:
        model_glob = os.path.join(loc_temp, uuid + "*")
        for fn in glob(model_glob):
            os.remove(fn)


def _sigmoid(x):
    """
    x : array-like
    output : array-like
    """
    return 1.0 / (1.0 + np.exp(-x))


def _validate_params(max_leaf,
                     test_interval,
                     algorithm,
                     loss,
                     reg_depth,
                     l2,
                     sl2,
                     normalize,
                     min_samples_leaf,
                     n_iter,
                     n_tree_search,
                     opt_interval,
                     learning_rate,
                     verbose,
                     calc_prob="Sigmoid"):
    if not isinstance(max_leaf, (numbers.Integral, np.integer)):
        raise ValueError("max_leaf must be an integer, got {0}.".format(type(max_leaf)))
    elif max_leaf <= 0:
        raise ValueError("max_leaf must be greater than 0 but was %r." % max_leaf)

    if not isinstance(test_interval, (numbers.Integral, np.integer)):
        raise ValueError("test_interval must be an integer, got {0}.".format(type(test_interval)))
    elif test_interval <= 0:
        raise ValueError("test_interval must be greater than 0 but was %r." % test_interval)

    if not isinstance(algorithm, six.string_types):
        raise ValueError("algorithm must be a string, got {0}.".format(type(algorithm)))
    elif algorithm not in _ALGORITHMS:
        raise ValueError("algorithm must be 'RGF' or 'RGF_Opt' or 'RGF_Sib' but was %r." % algorithm)

    if not isinstance(loss, six.string_types):
        raise ValueError("loss must be a string, got {0}.".format(type(loss)))
    elif loss not in _LOSSES:
        raise ValueError("loss must be 'LS' or 'Expo' or 'Log' but was %r." % loss)

    if not isinstance(reg_depth, (numbers.Integral, np.integer, _FLOATS)):
        raise ValueError("test_interval must be an integer or float, got {0}.".format(type(reg_depth)))
    elif reg_depth < 1:
        raise ValueError("reg_depth must be no smaller than 1.0 but was %r." % reg_depth)

    if not isinstance(l2, _FLOATS):
        raise ValueError("l2 must be a float, got {0}.".format(type(l2)))
    elif l2 < 0:
        raise ValueError("l2 must be no smaller than 0.0 but was %r." % l2)

    if not isinstance(sl2, (type(None), _FLOATS)):
        raise ValueError("sl2 must be a float or None, got {0}.".format(type(sl2)))
    elif sl2 is not None and sl2 < 0:
        raise ValueError("sl2 must be no smaller than 0.0 but was %r." % sl2)

    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean, got {0}.".format(type(normalize)))

    err_desc = "min_samples_leaf must be at least 1 or in (0, 0.5], got %r." % min_samples_leaf
    if isinstance(min_samples_leaf, (numbers.Integral, np.integer)):
        if min_samples_leaf < 1:
            raise ValueError(err_desc)
    elif isinstance(min_samples_leaf, _FLOATS):
        if not 0.0 < min_samples_leaf <= 0.5:
            raise ValueError(err_desc)
    else:
        raise ValueError("min_samples_leaf must be an integer or float, got {0}.".format(type(min_samples_leaf)))

    if not isinstance(n_iter, (type(None), numbers.Integral, np.integer)):
        raise ValueError("n_iter must be an integer or None, got {0}.".format(type(n_iter)))
    elif n_iter is not None and n_iter < 1:
        raise ValueError("n_iter must be no smaller than 1 but was %r." % n_iter)

    if not isinstance(n_tree_search, (numbers.Integral, np.integer)):
        raise ValueError("n_tree_search must be an integer, got {0}.".format(type(n_tree_search)))
    elif n_tree_search < 1:
        raise ValueError("n_tree_search must be no smaller than 1 but was %r." % n_tree_search)

    if not isinstance(opt_interval, (numbers.Integral, np.integer)):
        raise ValueError("opt_interval must be an integer, got {0}.".format(type(opt_interval)))
    elif opt_interval < 1:
        raise ValueError("opt_interval must be no smaller than 1 but was %r." % opt_interval)

    if not isinstance(learning_rate, _FLOATS):
        raise ValueError("learning_rate must be a float, got {0}.".format(type(learning_rate)))
    elif learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0 but was %r." % learning_rate)

    if not isinstance(verbose, (numbers.Integral, np.integer)):
        raise ValueError("verbose must be an integer, got {0}.".format(type(verbose)))
    elif verbose < 0:
        raise ValueError("verbose must be no smaller than 0 but was %r." % verbose)

    if not isinstance(calc_prob, six.string_types):
        raise ValueError("calc_prob must be a string, got {0}.".format(type(calc_prob)))
    elif calc_prob not in ("Sigmoid", "Softmax"):
        raise ValueError("calc_prob must be 'Sigmoid' or 'Softmax' but was %r." % calc_prob)


def _sparse_savetxt(filename, input_array):
    try:  # For Python 2.x
        from itertools import izip
        zip_func = izip
    except ImportError:  # For Python 3.x
        zip_func = zip
    input_array = input_array.tocsr().tocoo()
    n_row = input_array.shape[0]
    current_sample_row = 0
    line = []
    with open(filename, 'w') as fw:
        fw.write('sparse {0:d}\n'.format(input_array.shape[-1]))
        for i, j, v in zip_func(input_array.row, input_array.col, input_array.data):
            if i == current_sample_row:
                line.append('{0}:{1}'.format(j, v))
            else:
                fw.write(' '.join(line))
                fw.write('\n' * (i - current_sample_row))
                line = ['{0}:{1}'.format(j, v)]
                current_sample_row = i
        fw.write(' '.join(line))
        fw.write('\n' * (n_row - i))


class _AtomicCounter(object):
    def __init__(self):
        self.value = 0
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value


_COUNTER = _AtomicCounter()


class RGFClassifier(BaseEstimator, ClassifierMixin):
    """
    A Regularized Greedy Forest [1] classifier.

    Tuning parameters detailed instruction:
        http://tongzhang-ml.org/software/rgf/rgf1.2-guide.pdf

    Parameters
    ----------
    max_leaf : int, optional (default=1000)
        Training will be terminated when the number of
        leaf nodes in the forest reaches this value.

    test_interval : int, optional (default=100)
        Test interval in terms of the number of leaf nodes.

    algorithm : string ("RGF" or "RGF_Opt" or "RGF_Sib"), optional (default="RGF")
        Regularization algorithm.
        RGF: RGF with L2 regularization on leaf-only models.
        RGF Opt: RGF with min-penalty regularization.
        RGF Sib: RGF with min-penalty regularization with the sum-to-zero sibling constraints.

    loss : string ("LS" or "Expo" or "Log"), optional (default="Log")
        Loss function.

    reg_depth : float, optional (default=1.0)
        Must be no smaller than 1.0.
        Meant for being used with algorithm="RGF Opt"|"RGF Sib".
        A larger value penalizes deeper nodes more severely.

    l2 : float, optional (default=0.1)
        Used to control the degree of L2 regularization.

    sl2 : float or None, optional (default=None)
        Override L2 regularization parameter l2
        for the process of growing the forest.
        That is, if specified, the weight correction process uses l2
        and the forest growing process uses sl2.
        If None, no override takes place and
        l2 is used throughout training.

    normalize : boolean, optional (default=False)
        If True, training targets are normalized
        so that the average becomes zero.

    min_samples_leaf : int or float, optional (default=10)
        Minimum number of training data points in each leaf node.
        If int, then consider min_samples_leaf as the minimum number.
        If float, then min_samples_leaf is a percentage and
        ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

    n_iter : int or None, optional (default=None)
        Number of iterations of coordinate descent to optimize weights.
        If None, 10 is used for loss="LS" and 5 for loss="Expo"|"Log".

    n_tree_search : int, optional (default=1)
        Number of trees to be searched for the nodes to split.
        The most recently grown trees are searched first.

    opt_interval : int, optional (default=100)
        Weight optimization interval in terms of the number of leaf nodes.
        For example, by default, weight optimization is performed
        every time approximately 100 leaf nodes are newly added to the forest.

    learning_rate : float, optional (default=0.5)
        Step size of Newton updates used in coordinate descent to optimize weights.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    calc_prob : string ("Sigmoid" or "Softmax"), optional (default="Sigmoid")
        Method of probability calculation.

    Attributes:
    -----------
    estimators_ : list of binary classifiers
        The collection of fitted sub-estimators when `fit` is performed.

    classes_ : array of shape = [n_classes]
        The classes labels when `fit` is performed.

    n_classes_ : int
        The number of classes when `fit` is performed.

    n_features_ : int
        The number of features when `fit` is performed.

    fitted_ : boolean
        Indicates whether `fit` is performed.

    Reference
    ---------
    [1] Rie Johnson and Tong Zhang,
        Learning Nonlinear Functions Using Regularized Greedy Forest.
    """
    def __init__(self,
                 max_leaf=1000,
                 test_interval=100,
                 algorithm="RGF",
                 loss="Log",
                 reg_depth=1.0,
                 l2=0.1,
                 sl2=None,
                 normalize=False,
                 min_samples_leaf=10,
                 n_iter=None,
                 n_tree_search=1,
                 opt_interval=100,
                 learning_rate=0.5,
                 verbose=0,
                 calc_prob='Sigmoid'):
        self.max_leaf = max_leaf
        self.test_interval = test_interval
        self.algorithm = algorithm
        self.loss = loss
        self.reg_depth = reg_depth
        self.l2 = l2
        self.sl2 = sl2
        self.normalize = normalize
        self.min_samples_leaf = min_samples_leaf
        self.n_iter = n_iter
        self.n_tree_search = n_tree_search
        self.opt_interval = opt_interval
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.calc_prob = calc_prob
        self.fitted_ = False

    def fit(self, X, y, sample_weight=None):
        """
        Build a RGF Classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification).

        sample_weight : array-like, shape = [n_samples] or None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns self.
        """
        _validate_params(**self.get_params())

        if self.sl2 is None:
            self.sl2_ = self.l2
        else:
            self.sl2_ = self.sl2

        if isinstance(self.min_samples_leaf, _FLOATS):
            self.min_samples_leaf_ = ceil(self.min_samples_leaf * n_samples)
        else:
            self.min_samples_leaf_ = self.min_samples_leaf

        if self.n_iter is None:
            if self.loss == "LS":
                self.n_iter_ = 10
            else:
                self.n_iter_ = 5
        else:
            self.n_iter_ = self.n_iter

        X, y = check_X_y(X, y, accept_sparse=True)
        n_samples, self.n_features_ = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            if (sample_weight <= 0).any():
                raise ValueError("Sample weights must be positive.")
        check_consistent_length(X, y, sample_weight)
        check_classification_targets(y)

        self.classes_ = sorted(np.unique(y))
        self.n_classes_ = len(self.classes_)
        self._classes_map = {}
        if self.n_classes_ == 2:
            self._classes_map[0] = self.classes_[0]
            self._classes_map[1] = self.classes_[1]
            self.estimators_ = [None]
            y = (y == self.classes_[0]).astype(int)
            self.estimators_[0] = _RGFBinaryClassifier(max_leaf=self.max_leaf,
                                                       test_interval=self.test_interval,
                                                       algorithm=self.algorithm,
                                                       loss=self.loss,
                                                       reg_depth=self.reg_depth,
                                                       l2=self.l2,
                                                       sl2=self.sl2_,
                                                       normalize=self.normalize,
                                                       min_samples_leaf=self.min_samples_leaf_,
                                                       n_iter=self.n_iter_,
                                                       n_tree_search=self.n_tree_search,
                                                       opt_interval=self.opt_interval,
                                                       learning_rate=self.learning_rate,
                                                       verbose=self.verbose)
            self.estimators_[0].fit(X, y, sample_weight)
        elif self.n_classes_ > 2:
            self.estimators_ = [None] * self.n_classes_
            for i, cls_num in enumerate(self.classes_):
                self._classes_map[i] = cls_num
                y_one_or_rest = (y == cls_num).astype(int)
                self.estimators_[i] = _RGFBinaryClassifier(max_leaf=self.max_leaf,
                                                           test_interval=self.test_interval,
                                                           algorithm=self.algorithm,
                                                           loss=self.loss,
                                                           reg_depth=self.reg_depth,
                                                           l2=self.l2,
                                                           sl2=self.sl2_,
                                                           normalize=self.normalize,
                                                           min_samples_leaf=self.min_samples_leaf_,
                                                           n_iter=self.n_iter_,
                                                           n_tree_search=self.n_tree_search,
                                                           opt_interval=self.opt_interval,
                                                           learning_rate=self.learning_rate,
                                                           verbose=self.verbose)
                self.estimators_[i].fit(X, y_one_or_rest, sample_weight)
        else:
            raise ValueError("Classifier can't predict when only one class is present.")
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes].
            The class probabilities of the input samples.
            The order of the classes corresponds to that in the attribute classes_.
        """
        if not self.fitted_:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")
        X = check_array(X, accept_sparse=True)
        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))
        if self.n_classes_ == 2:
            proba = self.estimators_[0].predict_proba(X)
            proba = _sigmoid(proba)
            proba = np.c_[proba, 1 - proba]
        else:
            proba = np.zeros((X.shape[0], self.n_classes_))
            for i, clf in enumerate(self.estimators_):
                class_proba = clf.predict_proba(X)
                proba[:, i] = class_proba

            # In honest I don't understand which is better
            # softmax or normalized sigmoid for calc probability.
            if self.calc_prob == "Sigmoid":
                proba = _sigmoid(proba)
                normalizer = proba.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba /= normalizer
            else:
                proba = softmax(proba)
        return proba

    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is computed.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return np.asarray(list(self._classes_map.values()))[np.searchsorted(list(self._classes_map.keys()), y_pred)]


class _RGFBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    RGF Binary Classifier.
    Don't instantiate this class directly.
    RGFBinaryClassifier should be instantiated only by RGFClassifier.
    """
    def __init__(self,
                 max_leaf=500,
                 test_interval=100,
                 algorithm="RGF",
                 loss="Log",
                 reg_depth=1.0,
                 l2=0.1,
                 sl2=None,
                 normalize=False,
                 min_samples_leaf=10,
                 n_iter=None,
                 n_tree_search=1,
                 opt_interval=100,
                 learning_rate=0.5,
                 verbose=0):
        self.max_leaf = max_leaf
        self.test_interval = test_interval
        self.algorithm = algorithm
        self.loss = loss
        self.reg_depth = reg_depth
        self.l2 = l2
        self.sl2 = sl2
        self.normalize = normalize
        self.min_samples_leaf = min_samples_leaf
        self.n_iter = n_iter
        self.n_tree_search = n_tree_search
        self.opt_interval = opt_interval
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(_COUNTER.increment())
        _UUIDS.append(self._file_prefix)
        self.fitted_ = False

    def fit(self, X, y, sample_weight=None):
        train_x_loc = os.path.join(loc_temp, self._file_prefix + ".train.data.x")
        train_y_loc = os.path.join(loc_temp, self._file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(loc_temp, self._file_prefix + ".train.data.weight")
        if isspmatrix(X):
            _sparse_savetxt(train_x_loc, X)
        else:
            np.savetxt(train_x_loc, X, delimiter=' ', fmt="%s")

        # Convert 1 to 1, 0 to -1
        y = 2 * y - 1
        np.savetxt(train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        # Format train command
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        if self.normalize:
            params.append("NormalizeTarget")
        params.append("train_x_fn=%s" % train_x_loc)
        params.append("train_y_fn=%s" % train_y_loc)
        params.append("algorithm=%s" % self.algorithm)
        params.append("loss=%s" % self.loss)
        params.append("max_leaf_forest=%s" % self.max_leaf)
        params.append("test_interval=%s" % self.test_interval)
        params.append("reg_L2=%s" % self.l2)
        params.append("reg_sL2=%s" % self.sl2)
        params.append("reg_depth=%s" % self.reg_depth)
        params.append("min_pop=%s" % self.min_samples_leaf)
        params.append("num_iteration_opt=%s" % self.n_iter)
        params.append("num_tree_search=%s" % self.n_tree_search)
        params.append("opt_interval=%s" % self.opt_interval)
        params.append("opt_stepsize=%s" % self.learning_rate)
        params.append("model_fn_prefix=%s" % os.path.join(loc_temp, self._file_prefix + ".model"))
        params.append("train_w_fn=%s" % train_weight_loc)

        cmd = (loc_exec, "train", ",".join(params))

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        self.fitted_ = True
        return self

    def predict_proba(self, X):
        if not self.fitted_:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        test_x_loc = os.path.join(loc_temp, self._file_prefix + ".test.data.x")
        if isspmatrix(X):
            _sparse_savetxt(test_x_loc, X)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        # Find latest model location
        model_glob = os.path.join(loc_temp, self._file_prefix + ".model*")
        model_files = glob(model_glob)
        if not model_files:
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(loc_temp))
        latest_model_loc = sorted(model_files, reverse=True)[0]

        # Format test command
        pred_loc = os.path.join(loc_temp, self._file_prefix + ".predictions.txt")
        params = []
        params.append("test_x_fn=%s" % test_x_loc)
        params.append("prediction_fn=%s" % pred_loc)
        params.append("model_fn=%s" % latest_model_loc)

        cmd = (loc_exec, "predict", ",".join(params))

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        y_pred = np.loadtxt(pred_loc)
        return y_pred


class RGFRegressor(BaseEstimator, RegressorMixin):
    """
    A Regularized Greedy Forest [1] regressor.

    Tuning parameters detailed instruction:
        http://tongzhang-ml.org/software/rgf/rgf1.2-guide.pdf

    Parameters
    ----------
    max_leaf : int, optional (default=500)
        Training will be terminated when the number of
        leaf nodes in the forest reaches this value.

    test_interval : int, optional (default=100)
        Test interval in terms of the number of leaf nodes.

    algorithm : string ("RGF" or "RGF_Opt" or "RGF_Sib"), optional (default="RGF")
        Regularization algorithm.
        RGF: RGF with L2 regularization on leaf-only models.
        RGF Opt: RGF with min-penalty regularization.
        RGF Sib: RGF with min-penalty regularization with the sum-to-zero sibling constraints.

    loss : string ("LS" or "Expo" or "Log"), optional (default="LS")
        Loss function.

    reg_depth : float, optional (default=1.0)
        Must be no smaller than 1.0.
        Meant for being used with algorithm="RGF Opt"|"RGF Sib".
        A larger value penalizes deeper nodes more severely.

    l2 : float, optional (default=0.1)
        Used to control the degree of L2 regularization.

    sl2 : float or None, optional (default=None)
        Override L2 regularization parameter l2
        for the process of growing the forest.
        That is, if specified, the weight correction process uses l2
        and the forest growing process uses sl2.
        If None, no override takes place and
        l2 is used throughout training.

    normalize : boolean, optional (default=True)
        If True, training targets are normalized
        so that the average becomes zero.

    min_samples_leaf : int or float, optional (default=10)
        Minimum number of training data points in each leaf node.
        If int, then consider min_samples_leaf as the minimum number.
        If float, then min_samples_leaf is a percentage and
        ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

    n_iter : int or None, optional (default=None)
        Number of iterations of coordinate descent to optimize weights.
        If None, 10 is used for loss="LS" and 5 for loss="Expo"|"Log".

    n_tree_search : int, optional (default=1)
        Number of trees to be searched for the nodes to split.
        The most recently grown trees are searched first.

    opt_interval : int, optional (default=100)
        Weight optimization interval in terms of the number of leaf nodes.
        For example, by default, weight optimization is performed
        every time approximately 100 leaf nodes are newly added to the forest.

    learning_rate : float, optional (default=0.5)
        Step size of Newton updates used in coordinate descent to optimize weights.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes:
    -----------
    n_features_ : int
        The number of features when `fit` is performed.

    fitted_ : boolean
        Indicates whether `fit` is performed.

    Reference
    ---------
    [1] Rie Johnson and Tong Zhang,
        Learning Nonlinear Functions Using Regularized Greedy Forest.
    """
    def __init__(self,
                 max_leaf=500,
                 test_interval=100,
                 algorithm="RGF",
                 loss="LS",
                 reg_depth=1.0,
                 l2=0.1,
                 sl2=None,
                 normalize=True,
                 min_samples_leaf=10,
                 n_iter=None,
                 n_tree_search=1,
                 opt_interval=100,
                 learning_rate=0.5,
                 verbose=0):
        self.max_leaf = max_leaf
        self.test_interval = test_interval
        self.algorithm = algorithm
        self.loss = loss
        self.reg_depth = reg_depth
        self.l2 = l2
        self.sl2 = sl2
        self.normalize = normalize
        self.min_samples_leaf = min_samples_leaf
        self.n_iter = n_iter
        self.n_tree_search = n_tree_search
        self.opt_interval = opt_interval
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(_COUNTER.increment())
        _UUIDS.append(self._file_prefix)
        self.fitted_ = False

    def fit(self, X, y, sample_weight=None):
        """
        Build a RGF Regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (real numbers in regression).

        sample_weight : array-like, shape = [n_samples] or None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns self.
        """
        _validate_params(**self.get_params())

        if self.sl2 is None:
            self.sl2_ = self.l2
        else:
            self.sl2_ = self.sl2

        if isinstance(self.min_samples_leaf, _FLOATS):
            self.min_samples_leaf_ = ceil(self.min_samples_leaf * n_samples)
        else:
            self.min_samples_leaf_ = self.min_samples_leaf

        if self.n_iter is None:
            if self.loss == "LS":
                self.n_iter_ = 10
            else:
                self.n_iter_ = 5
        else:
            self.n_iter_ = self.n_iter

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=False, y_numeric=True)
        n_samples, self.n_features_ = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            if (sample_weight <= 0).any():
                raise ValueError("Sample weights must be positive.")
        check_consistent_length(X, y, sample_weight)

        train_x_loc = os.path.join(loc_temp, self._file_prefix + ".train.data.x")
        train_y_loc = os.path.join(loc_temp, self._file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(loc_temp, self._file_prefix + ".train.data.weight")
        if isspmatrix(X):
            _sparse_savetxt(train_x_loc, X)
        else:
            np.savetxt(train_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        # Format train command
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        if self.normalize:
            params.append("NormalizeTarget")
        params.append("train_x_fn=%s" % train_x_loc)
        params.append("train_y_fn=%s" % train_y_loc)
        params.append("algorithm=%s" % self.algorithm)
        params.append("loss=%s" % self.loss)
        params.append("max_leaf_forest=%s" % self.max_leaf)
        params.append("test_interval=%s" % self.test_interval)
        params.append("reg_L2=%s" % self.l2)
        params.append("reg_sL2=%s" % self.sl2_)
        params.append("reg_depth=%s" % self.reg_depth)
        params.append("min_pop=%s" % self.min_samples_leaf_)
        params.append("num_iteration_opt=%s" % self.n_iter_)
        params.append("num_tree_search=%s" % self.n_tree_search)
        params.append("opt_interval=%s" % self.opt_interval)
        params.append("opt_stepsize=%s" % self.learning_rate)
        params.append("model_fn_prefix=%s" % os.path.join(loc_temp, self._file_prefix + ".model"))
        params.append("train_w_fn=%s" % train_weight_loc)

        cmd = (loc_exec, "train", ",".join(params))

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        if not self.fitted_:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        X = check_array(X, accept_sparse=True)
        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        test_x_loc = os.path.join(loc_temp, self._file_prefix + ".test.data.x")
        if isspmatrix(X):
            _sparse_savetxt(test_x_loc, X)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        # Find latest model location
        model_glob = os.path.join(loc_temp, self._file_prefix + ".model*")
        model_files = glob(model_glob)
        if not model_files:
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(loc_temp))
        latest_model_loc = sorted(model_files, reverse=True)[0]

        # Format test command
        pred_loc = os.path.join(loc_temp, self._file_prefix + ".predictions.txt")
        params = []
        params.append("test_x_fn=%s" % test_x_loc)
        params.append("prediction_fn=%s" % pred_loc)
        params.append("model_fn=%s" % latest_model_loc)

        cmd = (loc_exec, "predict", ",".join(params))

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        y_pred = np.loadtxt(pred_loc)
        return y_pred
