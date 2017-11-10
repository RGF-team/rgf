from __future__ import absolute_import

__all__ = ('RGFClassifier', 'RGFRegressor')

from glob import glob
from math import ceil
from threading import Lock
from uuid import uuid4
import atexit
import codecs
import numbers
import os
import platform
import subprocess

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_X_y, column_or_1d

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()

_NOT_FITTED_ERROR_DESC = "Estimator not fitted, call `fit` before exploiting the model."
_ALGORITHMS = ("RGF", "RGF_Opt", "RGF_Sib")
_LOSSES = ("LS", "Expo", "Log")
_FLOATS = (float, np.float, np.float16, np.float32, np.float64, np.double)
_SYSTEM = platform.system()
_UUIDS = []


def _get_paths():
    config = six.moves.configparser.RawConfigParser()
    path = os.path.join(os.path.expanduser('~'), '.rgfrc')

    try:
        with codecs.open(path, 'r', 'utf-8') as cfg:
            with six.StringIO(cfg.read()) as strIO:
                config.readfp(strIO)
    except six.moves.configparser.MissingSectionHeaderError:
        with codecs.open(path, 'r', 'utf-8') as cfg:
            with six.StringIO('[glob]\n' + cfg.read()) as strIO:
                config.readfp(strIO)
    except Exception:
        pass

    if _SYSTEM in ('Windows', 'Microsoft'):
        try:
            exe = os.path.abspath(config.get(config.sections()[0], 'exe_location'))
        except Exception:
            exe = os.path.join(os.path.expanduser('~'), 'rgf.exe')
        try:
            temp = os.path.abspath(config.get(config.sections()[0], 'temp_location'))
        except Exception:
            temp = os.path.join(os.path.expanduser('~'), 'temp', 'rgf')
        def_exe = 'rgf.exe'
    else:  # Linux, Darwin (OS X), etc.
        try:
            exe = os.path.abspath(config.get(config.sections()[0], 'exe_location'))
        except Exception:
            exe = os.path.join(os.path.expanduser('~'), 'rgf')
        try:
            temp = os.path.abspath(config.get(config.sections()[0], 'temp_location'))
        except Exception:
            temp = os.path.join('/tmp', 'rgf')
        def_exe = 'rgf'

    return def_exe, exe, temp


_DEFAULT_EXE_PATH, _EXE_PATH, _TEMP_PATH = _get_paths()


if not os.path.isdir(_TEMP_PATH):
    os.makedirs(_TEMP_PATH)
if not os.access(_TEMP_PATH, os.W_OK):
    raise Exception("{0} is not writable directory. Please set "
                    "config flag 'temp_location' to writable directory".format(_TEMP_PATH))


def _is_executable_response(path):
    temp_x_loc = os.path.join(_TEMP_PATH, 'temp.train.data.x')
    temp_y_loc = os.path.join(_TEMP_PATH, 'temp.train.data.y')
    np.savetxt(temp_x_loc, [[1, 0, 1, 0], [0, 1, 0, 1]], delimiter=' ', fmt="%s")
    np.savetxt(temp_y_loc, [1, -1], delimiter=' ', fmt="%s")
    _UUIDS.append('temp')
    params = []
    params.append("train_x_fn=%s" % temp_x_loc)
    params.append("train_y_fn=%s" % temp_y_loc)
    params.append("model_fn_prefix=%s" % os.path.join(_TEMP_PATH, "temp.model"))
    params.append("reg_L2=%s" % 1)

    try:
        subprocess.check_output((path, "train", ",".join(params)))
        return True
    except Exception:
        return False


if _is_executable_response(_DEFAULT_EXE_PATH):
    _EXE_PATH = _DEFAULT_EXE_PATH
elif _is_executable_response(os.path.join(os.path.dirname(__file__), _DEFAULT_EXE_PATH)):
    _EXE_PATH = os.path.join(os.path.dirname(__file__), _DEFAULT_EXE_PATH)
elif not os.path.isfile(_EXE_PATH):
    raise Exception("{0} is not executable file. Please set "
                    "config flag 'exe_location' to RGF execution file.".format(_EXE_PATH))
elif not os.access(_EXE_PATH, os.X_OK):
    raise Exception("{0} cannot be accessed. Please set "
                    "config flag 'exe_location' to RGF execution file.".format(_EXE_PATH))
elif _is_executable_response(_EXE_PATH):
    pass
else:
    raise Exception("{0} does not exist or {1} is not in the "
                    "'PATH' variable.".format(_EXE_PATH, _DEFAULT_EXE_PATH))


@atexit.register
def _cleanup():
    if _UUIDS is not None:
        for uuid in _UUIDS:
            model_glob = os.path.join(_TEMP_PATH, uuid + "*")
            for fn in glob(model_glob):
                os.remove(fn)


def _sigmoid(x):
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
                     memory_policy,
                     calc_prob="sigmoid",
                     n_jobs=-1):
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

    if not isinstance(memory_policy, six.string_types):
        raise ValueError("memory_policy must be a string, got {0}.".format(type(memory_policy)))
    elif memory_policy not in ("conservative", "generous"):
        raise ValueError("memory_policy must be 'conservative' or 'generous' but was %r." % memory_policy)

    if not isinstance(calc_prob, six.string_types):
        raise ValueError("calc_prob must be a string, got {0}.".format(type(calc_prob)))
    elif calc_prob not in ("sigmoid", "softmax"):
        raise ValueError("calc_prob must be 'sigmoid' or 'softmax' but was %r." % calc_prob)

    if not isinstance(n_jobs, (numbers.Integral, np.integer)):
        raise ValueError("n_jobs must be an integer, got {0}.".format(type(n_jobs)))


def _sparse_savetxt(filename, input_array):
    zip_func = six.moves.zip
    if sp.isspmatrix_csr(input_array):
        input_array = input_array.tocoo()
    elif not sp.isspmatrix_coo(input_array):
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


def _fit_ovr_binary(binary_clf, X, y, sample_weight):
    return binary_clf.fit(X, y, sample_weight)


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
        https://github.com/fukatani/rgf_python/blob/master/include/rgf/rgf1.2-guide.pdf

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
        LS: Square loss.
        Expo: Exponential loss.
        Log: Logistic loss.

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

    calc_prob : string ("sigmoid" or "softmax"), optional (default="sigmoid")
        Method of probability calculation.

    n_jobs : integer, optional (default=-1)
        The number of jobs to use for the computation.
        If 1 is given, no parallel computing code is used at all.
        If -1 all CPUs are used.
        For n_jobs = -2, all CPUs but one are used.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

    memory_policy : string ("conservative" or "generous"), optional (default="generous")
        Memory using policy.
        Generous: it runs faster using more memory by keeping the sorted orders
        of the features on memory for reuse.
        Conservative: it uses less memory at the expense of longer runtime. Try only when
        with default value it uses too much memory.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

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

    sl2_ : float
        The concrete regularization value for the process of growing the forest
        used in model building process.

    min_samples_leaf_ : int
        Minimum number of training data points in each leaf node
        used in model building process.

    n_iter_ : int
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.

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
                 calc_prob="sigmoid",
                 n_jobs=-1,
                 memory_policy="generous",
                 verbose=0):
        self.max_leaf = max_leaf
        self.test_interval = test_interval
        self.algorithm = algorithm
        self.loss = loss
        self.reg_depth = reg_depth
        self.l2 = l2
        self.sl2 = sl2
        self._sl2 = None
        self.normalize = normalize
        self.min_samples_leaf = min_samples_leaf
        self._min_samples_leaf = None
        self.n_iter = n_iter
        self._n_iter = None
        self.n_tree_search = n_tree_search
        self.opt_interval = opt_interval
        self.learning_rate = learning_rate
        self.calc_prob = calc_prob
        self.n_jobs = n_jobs
        self.memory_policy = memory_policy
        self.verbose = verbose
        self._estimators = None
        self._classes = None
        self._n_classes = None
        self._n_features = None
        self._fitted = None

    @property
    def estimators_(self):
        """The collection of fitted sub-estimators when `fit` is performed."""
        if self._estimators is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._estimators

    @property
    def classes_(self):
        """The classes labels when `fit` is performed."""
        if self._classes is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._classes

    @property
    def n_classes_(self):
        """The number of classes when `fit` is performed."""
        if self._n_classes is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._n_classes

    @property
    def n_features_(self):
        """The number of features when `fit` is performed."""
        if self._n_features is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._n_features

    @property
    def fitted_(self):
        """Indicates whether `fit` is performed."""
        if self._fitted is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._fitted

    @property
    def sl2_(self):
        """
        The concrete regularization value for the process of growing the forest
        used in model building process.
        """
        if self._sl2 is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._sl2

    @property
    def min_samples_leaf_(self):
        """
        Minimum number of training data points in each leaf node
        used in model building process.
        """
        if self._min_samples_leaf is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._min_samples_leaf

    @property
    def n_iter_(self):
        """
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.
        """
        if self._n_iter is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._n_iter

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

        X, y = check_X_y(X, y, accept_sparse=True)
        n_samples, self._n_features = X.shape

        if self.sl2 is None:
            self._sl2 = self.l2
        else:
            self._sl2 = self.sl2

        if isinstance(self.min_samples_leaf, _FLOATS):
            self._min_samples_leaf = ceil(self.min_samples_leaf * n_samples)
        else:
            self._min_samples_leaf = self.min_samples_leaf

        if self.n_iter is None:
            if self.loss == "LS":
                self._n_iter = 10
            else:
                self._n_iter = 5
        else:
            self._n_iter = self.n_iter

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            if (sample_weight <= 0).any():
                raise ValueError("Sample weights must be positive.")
        check_consistent_length(X, y, sample_weight)
        check_classification_targets(y)

        self._classes = sorted(np.unique(y))
        self._n_classes = len(self._classes)
        self._classes_map = {}

        params = dict(max_leaf=self.max_leaf,
                      test_interval=self.test_interval,
                      algorithm=self.algorithm,
                      loss=self.loss,
                      reg_depth=self.reg_depth,
                      l2=self.l2,
                      sl2=self._sl2,
                      normalize=self.normalize,
                      min_samples_leaf=self._min_samples_leaf,
                      n_iter=self._n_iter,
                      n_tree_search=self.n_tree_search,
                      opt_interval=self.opt_interval,
                      learning_rate=self.learning_rate,
                      memory_policy=self.memory_policy,
                      verbose=self.verbose)
        if self._n_classes == 2:
            self._classes_map[0] = self._classes[0]
            self._classes_map[1] = self._classes[1]
            self._estimators = [None]
            y = (y == self._classes[0]).astype(int)
            self._estimators[0] = _RGFBinaryClassifier(**params)
            self._estimators[0].fit(X, y, sample_weight)
        elif self._n_classes > 2:
            if sp.isspmatrix_dok(X):
                X = X.tocsr().tocoo()  # Fix to avoid scipy 7699 issue
            self._estimators = [None] * self._n_classes
            ovr_list = [None] * self._n_classes
            for i, cls_num in enumerate(self._classes):
                self._classes_map[i] = cls_num
                ovr_list[i] = (y == cls_num).astype(int)
                self._estimators[i] = _RGFBinaryClassifier(**params)
            self._estimators = Parallel(n_jobs=self.n_jobs)(delayed(_fit_ovr_binary)(self._estimators[i],
                                                                                     X,
                                                                                     ovr_list[i],
                                                                                     sample_weight)
                                                            for i in range(self._n_classes))
        else:
            raise ValueError("Classifier can't predict when only one class is present.")

        self._fitted = True
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
        if self._fitted is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        X = check_array(X, accept_sparse=True)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))
        if self._n_classes == 2:
            y = self._estimators[0].predict_proba(X)
            y = _sigmoid(y)
            y = np.c_[y, 1 - y]
        else:
            y = np.zeros((X.shape[0], self._n_classes))
            for i, clf in enumerate(self._estimators):
                class_proba = clf.predict_proba(X)
                y[:, i] = class_proba

            # In honest, I don't understand which is better
            # softmax or normalized sigmoid for calc probability.
            if self.calc_prob == "sigmoid":
                y = _sigmoid(y)
                normalizer = np.sum(y, axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                y /= normalizer
            else:
                y = softmax(y)
        return y

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
        y = self.predict_proba(X)
        y = np.argmax(y, axis=1)
        return np.asarray(list(self._classes_map.values()))[np.searchsorted(list(self._classes_map.keys()), y)]


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
                 memory_policy="generous",
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
        self.memory_policy = memory_policy
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(_COUNTER.increment())
        _UUIDS.append(self._file_prefix)
        self._fitted = None
        self._latest_model_loc = None

    def fit(self, X, y, sample_weight):
        train_x_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".train.data.x")
        train_y_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".train.data.weight")
        if sp.isspmatrix(X):
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
        if self.verbose > 5:
            params.append("Verbose_opt")  # Add some info on weight optimization
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
        params.append("memory_policy=%s" % self.memory_policy.title())
        params.append("model_fn_prefix=%s" % os.path.join(_TEMP_PATH, self._file_prefix + ".model"))
        params.append("train_w_fn=%s" % train_weight_loc)

        cmd = (_EXE_PATH, "train", ",".join(params))

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        self._fitted = True
        # Find latest model location
        model_glob = os.path.join(_TEMP_PATH, self._file_prefix + ".model*")
        model_files = glob(model_glob)
        if not model_files:
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(_TEMP_PATH))
        self._latest_model_loc = sorted(model_files, reverse=True)[0]
        return self

    def predict_proba(self, X):
        if self._fitted is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        if not os.path.isfile(self._latest_model_loc):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(_TEMP_PATH))

        test_x_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".test.data.x")
        if sp.isspmatrix(X):
            _sparse_savetxt(test_x_loc, X)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        # Format test command
        pred_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".predictions.txt")
        params = []
        params.append("test_x_fn=%s" % test_x_loc)
        params.append("prediction_fn=%s" % pred_loc)
        params.append("model_fn=%s" % self._latest_model_loc)

        cmd = (_EXE_PATH, "predict", ",".join(params))

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        return np.loadtxt(pred_loc)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._fitted:
            with open(self._latest_model_loc, 'rb') as fr:
                state["model"] = fr.read()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._fitted:
            with open(self._latest_model_loc, 'wb') as fw:
                fw.write(self.__dict__["model"])
            del self.__dict__["model"]


class RGFRegressor(BaseEstimator, RegressorMixin):
    """
    A Regularized Greedy Forest [1] regressor.

    Tuning parameters detailed instruction:
        https://github.com/fukatani/rgf_python/blob/master/include/rgf/rgf1.2-guide.pdf

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

    memory_policy : string ("conservative" or "generous"), optional (default="generous")
        Memory using policy.
        Generous: it runs faster using more memory by keeping the sorted orders
        of the features on memory for reuse.
        Conservative: it uses less memory at the expense of longer runtime. Try only when
        with default value it uses too much memory.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes:
    -----------
    n_features_ : int
        The number of features when `fit` is performed.

    fitted_ : boolean
        Indicates whether `fit` is performed.

    sl2_ : float
        The concrete regularization value for the process of growing the forest
        when `fit` is performed.

    min_samples_leaf_ : int
        Minimum number of training data points in each leaf node
        used in model building process.

    n_iter_ : int
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.

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
                 memory_policy="generous",
                 verbose=0):
        self.max_leaf = max_leaf
        self.test_interval = test_interval
        self.algorithm = algorithm
        self.loss = loss
        self.reg_depth = reg_depth
        self.l2 = l2
        self.sl2 = sl2
        self._sl2 = None
        self.normalize = normalize
        self.min_samples_leaf = min_samples_leaf
        self._min_samples_leaf = None
        self.n_iter = n_iter
        self._n_iter = None
        self.n_tree_search = n_tree_search
        self.opt_interval = opt_interval
        self.learning_rate = learning_rate
        self.memory_policy = memory_policy
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(_COUNTER.increment())
        _UUIDS.append(self._file_prefix)
        self._n_features = None
        self._fitted = None
        self._latest_model_loc = None

    @property
    def n_features_(self):
        """The number of features when `fit` is performed."""
        if self._n_features is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._n_features

    @property
    def fitted_(self):
        """Indicates whether `fit` is performed."""
        if self._fitted is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._fitted

    @property
    def sl2_(self):
        """
        The concrete regularization value for the process of growing the forest
        used in model building process.
        """
        if self._sl2 is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._sl2

    @property
    def min_samples_leaf_(self):
        """
        Minimum number of training data points in each leaf node
        used in model building process.
        """
        if self._min_samples_leaf is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._min_samples_leaf

    @property
    def n_iter_(self):
        """
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.
        """
        if self._n_iter is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)
        else:
            return self._n_iter

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

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=False, y_numeric=True)
        n_samples, self._n_features = X.shape

        if self.sl2 is None:
            self._sl2 = self.l2
        else:
            self._sl2 = self.sl2

        if isinstance(self.min_samples_leaf, _FLOATS):
            self._min_samples_leaf = ceil(self.min_samples_leaf * n_samples)
        else:
            self._min_samples_leaf = self.min_samples_leaf

        if self.n_iter is None:
            if self.loss == "LS":
                self._n_iter = 10
            else:
                self._n_iter = 5
        else:
            self._n_iter = self.n_iter

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            if (sample_weight <= 0).any():
                raise ValueError("Sample weights must be positive.")
        check_consistent_length(X, y, sample_weight)

        train_x_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".train.data.x")
        train_y_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".train.data.weight")
        if sp.isspmatrix(X):
            _sparse_savetxt(train_x_loc, X)
        else:
            np.savetxt(train_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        # Format train command
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        if self.verbose > 5:
            params.append("Verbose_opt")  # Add some info on weight optimization
        if self.normalize:
            params.append("NormalizeTarget")
        params.append("train_x_fn=%s" % train_x_loc)
        params.append("train_y_fn=%s" % train_y_loc)
        params.append("algorithm=%s" % self.algorithm)
        params.append("loss=%s" % self.loss)
        params.append("max_leaf_forest=%s" % self.max_leaf)
        params.append("test_interval=%s" % self.test_interval)
        params.append("reg_L2=%s" % self.l2)
        params.append("reg_sL2=%s" % self._sl2)
        params.append("reg_depth=%s" % self.reg_depth)
        params.append("min_pop=%s" % self._min_samples_leaf)
        params.append("num_iteration_opt=%s" % self._n_iter)
        params.append("num_tree_search=%s" % self.n_tree_search)
        params.append("opt_interval=%s" % self.opt_interval)
        params.append("opt_stepsize=%s" % self.learning_rate)
        params.append("memory_policy=%s" % self.memory_policy.title())
        params.append("model_fn_prefix=%s" % os.path.join(_TEMP_PATH, self._file_prefix + ".model"))
        params.append("train_w_fn=%s" % train_weight_loc)

        cmd = (_EXE_PATH, "train", ",".join(params))

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        self._fitted = True

        # Find latest model location
        model_glob = os.path.join(_TEMP_PATH, self._file_prefix + ".model*")
        model_files = glob(model_glob)
        if not model_files:
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(_TEMP_PATH))
        self._latest_model_loc = sorted(model_files, reverse=True)[0]
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
        if self._fitted is None:
            raise NotFittedError(_NOT_FITTED_ERROR_DESC)

        X = check_array(X, accept_sparse=True)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))

        test_x_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".test.data.x")
        if sp.isspmatrix(X):
            _sparse_savetxt(test_x_loc, X)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        if not os.path.isfile(self._latest_model_loc):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(_TEMP_PATH))

        # Format test command
        pred_loc = os.path.join(_TEMP_PATH, self._file_prefix + ".predictions.txt")
        params = []
        params.append("test_x_fn=%s" % test_x_loc)
        params.append("prediction_fn=%s" % pred_loc)
        params.append("model_fn=%s" % self._latest_model_loc)

        cmd = (_EXE_PATH, "predict", ",".join(params))

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        y_pred = np.loadtxt(pred_loc)
        return y_pred

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._fitted:
            with open(self._latest_model_loc, 'rb') as fr:
                state["model"] = fr.read()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._fitted:
            with open(self._latest_model_loc, 'wb') as fw:
                fw.write(self.__dict__["model"])
            del self.__dict__["model"]
