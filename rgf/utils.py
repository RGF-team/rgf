from __future__ import absolute_import


import atexit
import codecs
import glob
import os
import platform
import stat
import subprocess
from threading import Lock
from uuid import uuid4

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.externals import six
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_array


with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()

_NOT_FITTED_ERROR_DESC = "Estimator not fitted, call `fit` before exploiting the model."
_SYSTEM = platform.system()
UUIDS = []
_FASTRGF_AVAILABLE = False


@atexit.register
def cleanup():
    for uuid in UUIDS:
        cleanup_partial(uuid)


def cleanup_partial(uuid, remove_from_list=False):
    n_removed_files = 0
    if uuid in UUIDS:
        model_glob = os.path.join(_TEMP_PATH, uuid + "*")
        for fn in glob.glob(model_glob):
            os.remove(fn)
            n_removed_files += 1
        if remove_from_list:
            UUIDS.remove(uuid)
    return n_removed_files


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
            rgf_exe = os.path.abspath(config.get(config.sections()[0], 'exe_location'))
        except Exception:
            rgf_exe = os.path.join(os.path.expanduser('~'), 'rgf.exe')
        try:
            fast_rgf_path = os.path.abspath(config.get(config.sections()[0], 'fastrgf_location'))
        except Exception:
            fast_rgf_path = os.path.expanduser('~')
        try:
            temp = os.path.abspath(config.get(config.sections()[0], 'temp_location'))
        except Exception:
            temp = os.path.join(os.path.expanduser('~'), 'temp', 'rgf')
        def_exe = 'rgf.exe'
    else:  # Linux, Darwin (OS X), etc.
        try:
            rgf_exe = os.path.abspath(config.get(config.sections()[0], 'exe_location'))
        except Exception:
            rgf_exe = os.path.join(os.path.expanduser('~'), 'rgf')
        try:
            fast_rgf_path = os.path.abspath(config.get(config.sections()[0], 'fastrgf_location'))
        except Exception:
            fast_rgf_path = os.path.expanduser('~')
        try:
            temp = os.path.abspath(config.get(config.sections()[0], 'temp_location'))
        except Exception:
            temp = os.path.join('/tmp', 'rgf')
        def_exe = 'rgf'

    return def_exe, rgf_exe, fast_rgf_path, temp


_DEFAULT_EXE_PATH, _EXE_PATH, _FASTRGF_PATH, _TEMP_PATH = _get_paths()


if not os.path.isdir(_TEMP_PATH):
    os.makedirs(_TEMP_PATH)
if not os.access(_TEMP_PATH, os.W_OK):
    raise Exception("{0} is not writable directory. Please set "
                    "config flag 'temp_location' to writable directory".format(_TEMP_PATH))


def _is_rgf_executable(path):
    temp_x_loc = os.path.join(_TEMP_PATH, 'temp.train.data.x')
    temp_y_loc = os.path.join(_TEMP_PATH, 'temp.train.data.y')
    np.savetxt(temp_x_loc, [[1, 0, 1, 0], [0, 1, 0, 1]], delimiter=' ', fmt="%s")
    np.savetxt(temp_y_loc, [1, -1], delimiter=' ', fmt="%s")
    UUIDS.append('temp')
    params = []
    params.append("train_x_fn=%s" % temp_x_loc)
    params.append("train_y_fn=%s" % temp_y_loc)
    params.append("model_fn_prefix=%s" % os.path.join(_TEMP_PATH, "temp.model"))
    params.append("reg_L2=%s" % 1)
    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass
    try:
        subprocess.check_output((path, "train", ",".join(params)))
        return True
    except Exception:
        return False


def _is_fastrgf_executable(path):
    train_exec = os.path.join(path, "forest_train")
    try:
        subprocess.check_output([train_exec, "--help"])
    except Exception:
        return False
    pred_exec = os.path.join(path, "forest_predict")
    try:
        subprocess.check_output([pred_exec, "--help"])
    except Exception:
        return False
    return True


if _is_rgf_executable(_DEFAULT_EXE_PATH):
    _EXE_PATH = _DEFAULT_EXE_PATH
elif _is_rgf_executable(os.path.join(os.path.dirname(__file__), _DEFAULT_EXE_PATH)):
    _EXE_PATH = os.path.join(os.path.dirname(__file__), _DEFAULT_EXE_PATH)
elif not os.path.isfile(_EXE_PATH):
    raise Exception("{0} is not executable file. Please set "
                    "config flag 'exe_location' to RGF execution file.".format(_EXE_PATH))
elif not os.access(_EXE_PATH, os.X_OK):
    raise Exception("{0} cannot be accessed. Please set "
                    "config flag 'exe_location' to RGF execution file.".format(_EXE_PATH))
elif _is_rgf_executable(_EXE_PATH):
    pass
else:
    raise Exception("{0} does not exist or {1} is not in the "
                    "'PATH' variable.".format(_EXE_PATH, _DEFAULT_EXE_PATH))

_FASTRGF_AVAILABLE = _is_fastrgf_executable(_FASTRGF_PATH)


def fastrgf_available():
    return _FASTRGF_AVAILABLE


def get_temp_path():
    return _TEMP_PATH


def get_exe_path():
    return _EXE_PATH


def get_fastrgf_path():
    return _FASTRGF_PATH


class _AtomicCounter(object):
    def __init__(self):
        self.value = 0
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value


COUNTER = _AtomicCounter()


def not_fitted_error_desc():
    return "Estimator not fitted, call `fit` before exploiting the model."


def sparse_savetxt(filename, input_array, including_header=True):
    zip_func = six.moves.zip
    if sp.isspmatrix_csr(input_array):
        input_array = input_array.tocoo()
    elif not sp.isspmatrix_coo(input_array):
        input_array = input_array.tocsr().tocoo()
    n_row = input_array.shape[0]
    current_sample_row = 0
    line = []
    with open(filename, 'w') as fw:
        if including_header:
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fit_ovr_binary(binary_clf, X, y, sample_weight):
    return binary_clf.fit(X, y, sample_weight)


class RGFClassifierBase(BaseEstimator, ClassifierMixin):
    @property
    def estimators_(self):
        """The collection of fitted sub-estimators when `fit` is performed."""
        if self._estimators is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._estimators

    @property
    def classes_(self):
        """The classes labels when `fit` is performed."""
        if self._classes is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._classes

    @property
    def n_classes_(self):
        """The number of classes when `fit` is performed."""
        if self._n_classes is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._n_classes

    @property
    def n_features_(self):
        """The number of features when `fit` is performed."""
        if self._n_features is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._n_features

    @property
    def fitted_(self):
        """Indicates whether `fit` is performed."""
        if self._fitted is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._fitted

    @property
    def n_iter_(self):
        """
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.
        """
        if self._n_iter is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._n_iter

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
            raise NotFittedError(not_fitted_error_desc())
        X = check_array(X, accept_sparse=True)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))
        if self._n_classes == 2:
            y = self._estimators[0].predict_proba(X)
            y = sigmoid(y)
            y = np.c_[y, 1 - y]
        else:
            y = np.zeros((X.shape[0], self._n_classes))
            for i, clf in enumerate(self._estimators):
                class_proba = clf.predict_proba(X)
                y[:, i] = class_proba

            # In honest, I don't understand which is better
            # softmax or normalized sigmoid for calc probability.
            if self.calc_prob == "sigmoid":
                y = sigmoid(y)
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

    def cleanup(self):
        """
        Remove tempfiles used by this model.

        Returns
        -------
        n_removed_files : int
            Returns the number of removed files.
        """
        n_removed_files = 0
        if self._estimators is not None:
            for est in self._estimators:
                n_removed_files += cleanup_partial(est.file_prefix,
                                                   remove_from_list=True)

        # No more able to predict without refitting.
        self._fitted = None
        return n_removed_files


class RGFRegressorBase(BaseEstimator, RegressorMixin):
    @property
    def n_features_(self):
        """The number of features when `fit` is performed."""
        if self._n_features is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._n_features

    @property
    def fitted_(self):
        """Indicates whether `fit` is performed."""
        if self._fitted is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._fitted

    @property
    def n_iter_(self):
        """
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.
        """
        if self._n_iter is None:
            raise NotFittedError(not_fitted_error_desc())
        else:
            return self._n_iter

    def cleanup(self):
        """
        Remove tempfiles used by this model.

        Returns
        -------
        n_removed_files : int
            Returns the number of removed files.
        """
        # No more able to predict without refitting.
        self._fitted = None
        return cleanup_partial(self._file_prefix, remove_from_list=True)


class RGFBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    RGF Binary Classifier.
    Don't instantiate this class directly.
    This class should be instantiated only by RGFClassifier or FastRGFClassifier.
    """
    def __init__(self, fast_rgf=False, **kwargs):
        self.fast_rgf = fast_rgf
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_prefix = str(uuid4()) + str(COUNTER.increment())
        UUIDS.append(self.file_prefix)
        self.fitted = None

    def fit(self, X, y, sample_weight):
        train_x_loc = os.path.join(get_temp_path(), self.file_prefix + ".train.data.x")
        train_y_loc = os.path.join(get_temp_path(), self.file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(get_temp_path(), self.file_prefix + ".train.data.weight")
        model_file = os.path.join(get_temp_path(), self.file_prefix + ".model")

        if sp.isspmatrix(X):
            sparse_savetxt(train_x_loc, X, including_header=not self.fast_rgf)
        else:
            np.savetxt(train_x_loc, X, delimiter=' ', fmt="%s")

        # Convert 1 to 1, 0 to -1
        y = 2 * y - 1
        np.savetxt(train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        # Format train command
        params = []
        if self.fast_rgf:
            params.append("forest.ntrees=%s" % self.forest_ntrees)
            params.append("discretize.dense.lamL2=%s" % self.discretize_dense_lamL2)
            params.append("discretize.sparse.max_features=%s" % self.discretize_sparse_max_features)
            params.append("discretize.sparse.max_buckets=%s" % self.discretize_sparse_max_buckets)
            params.append("discretize.dense.max_buckets=%s" % self.discretize_dense_max_buckets)
            params.append("dtree.new_tree_gain_ratio=%s" % self.dtree_new_tree_gain_ratio)
            params.append("dtree.loss=%s" % self.dtree_loss)
            params.append("dtree.lamL1=%s" % self.dtree_lamL1)
            params.append("dtree.lamL2=%s" % self.dtree_lamL2)
            params.append("trn.x-file=%s" % train_x_loc)
            params.append("trn.y-file=%s" % train_y_loc)
            params.append("trn.w-file=%s" % train_weight_loc)
            if sp.isspmatrix(X):
                params.append("trn.x-file_format=x.sparse")
            params.append("trn.target=BINARY")
            params.append("set.nthreads=%s" % self.nthreads)
            params.append("set.verbose=%s" % self.verbose)
            params.append("model.save=%s" % model_file)

            cmd = [get_fastrgf_path() + "/forest_train"]
            cmd.extend(params)
        else:
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
            params.append("model_fn_prefix=%s" % model_file)
            params.append("train_w_fn=%s" % train_weight_loc)

            cmd = (get_exe_path(), "train", ",".join(params))

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        if not self.fast_rgf:
            # Find latest model location
            model_glob = os.path.join(get_temp_path(), self.file_prefix + ".model*")
            model_files = glob.glob(model_glob)
            if not model_files:
                raise Exception('Model learning result is not found in {0}. '
                                'Training is abnormally finished.'.format(get_temp_path()))
            self.model_file = sorted(model_files, reverse=True)[0]
        else:
            if not os.path.isfile(model_file):
                raise Exception('Model learning result is not found in {0}. '
                                'Training is abnormally finished.'.format(get_temp_path()))
            self.model_file = model_file

        self.fitted = True
							        
        return self

    def predict_proba(self, X):
        if self.fitted is None:
            raise NotFittedError(not_fitted_error_desc())
        if not os.path.isfile(self.model_file):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(get_temp_path()))

        test_x_loc = os.path.join(get_temp_path(), self.file_prefix + ".test.data.x")
        if sp.isspmatrix(X):
            sparse_savetxt(test_x_loc, X, including_header=not self.fast_rgf)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        pred_loc = os.path.join(get_temp_path(), self.file_prefix + ".predictions.txt")

        # Format test command
        params = []
        if self.fast_rgf:
            params.append("model.load=%s" % self.model_file)
            params.append("tst.x-file=%s" % test_x_loc)
            if sp.isspmatrix(X):
                params.append("tst.x-file_format=x.sparse")
            params.append("tst.target=REAL")
            params.append("tst.output-prediction=%s" % pred_loc)
            params.append("set.nthreads=%s" % self.nthreads)
            params.append("set.verbose=%s" % self.verbose)

            cmd = [get_fastrgf_path() + "/forest_predict"]
            cmd.extend(params)
        else:
            params.append("test_x_fn=%s" % test_x_loc)
            params.append("prediction_fn=%s" % pred_loc)
            params.append("model_fn=%s" % self.model_file)

            cmd = (get_exe_path(), "predict", ",".join(params))

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT).communicate()

        if self.verbose:
            for k in output:
                print(k)

        return np.loadtxt(pred_loc)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.fitted:
            with open(self.model_file, 'rb') as fr:
                state["model"] = fr.read()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.fitted:
            with open(self.model_file, 'wb') as fw:
                fw.write(self.__dict__["model"])
            del self.__dict__["model"]
