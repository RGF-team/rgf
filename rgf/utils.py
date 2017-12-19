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
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_X_y, column_or_1d


with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()

_NOT_FITTED_ERROR_DESC = "Estimator not fitted, call `fit` before exploiting the model."
_NOT_IMPLEMENTED_ERROR_DESC = "This method isn't implemented in base class."
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

    def fit(self, X, y, sample_weight=None):
        """
        Build a classifier from the training set (X, y).

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
        self._validate_params(self.get_params())
        self._set_params_with_dependencies()

        X, y = check_X_y(X, y, accept_sparse=True)
        n_samples, self._n_features = X.shape

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

        params = self._get_params()

        if self._n_classes == 2:
            self._classes_map[0] = self._classes[0]
            self._classes_map[1] = self._classes[1]
            self._estimators = [None]
            y = (y == self._classes[0]).astype(int)
            self._fit_binary_task(X, y, sample_weight, params)
        elif self._n_classes > 2:
            if sp.isspmatrix_dok(X):
                X = X.tocsr().tocoo()  # Fix to avoid scipy 7699 issue
            self._estimators = [None] * self._n_classes
            self._fit_multiclass_task(X, y, sample_weight, params)
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

    def _validate_params(self, params):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

    def _set_params_with_dependencies(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

    def _get_params(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

    def _fit_binary_task(self, X, y, sample_weight, params):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

    def _fit_multiclass_task(self, X, y, sample_weight, params):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)


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


class RGFBinaryClassifierBase(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_prefix = str(uuid4()) + str(COUNTER.increment())
        UUIDS.append(self.file_prefix)
        self.fitted = None

    def fit(self, X, y, sample_weight):
        self.train_x_loc = os.path.join(get_temp_path(), self.file_prefix + ".train.data.x")
        self.train_y_loc = os.path.join(get_temp_path(), self.file_prefix + ".train.data.y")
        self.train_weight_loc = os.path.join(get_temp_path(), self.file_prefix + ".train.data.weight")
        self.model_file_loc = os.path.join(get_temp_path(), self.file_prefix + ".model")

        if sp.isspmatrix(X):
            self.save_sparse_X(self.train_x_loc, X)
            self.is_sparse_train_X = True
        else:
            np.savetxt(self.train_x_loc, X, delimiter=' ', fmt="%s")
            self.is_sparse_train_X = False

        # Convert 1 to 1, 0 to -1
        y = 2 * y - 1
        np.savetxt(self.train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(self.train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        cmd = self.get_train_command()

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        self.find_model_file()

        self.fitted = True
					        
        return self

    def predict_proba(self, X):
        if self.fitted is None:
            raise NotFittedError(not_fitted_error_desc())
        if not os.path.isfile(self.model_file):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(get_temp_path()))

        self.test_x_loc = os.path.join(get_temp_path(), self.file_prefix + ".test.data.x")
        if sp.isspmatrix(X):
            self.save_sparse_X(self.test_x_loc, X)
            self.is_sparse_test_X = True
        else:
            np.savetxt(self.test_x_loc, X, delimiter=' ', fmt="%s")
            self.is_sparse_test_X = False

        self.pred_loc = os.path.join(get_temp_path(), self.file_prefix + ".predictions.txt")

        cmd = self.get_test_command()

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT).communicate()

        if self.verbose:
            for k in output:
                print(k)

        return np.loadtxt(self.pred_loc)

    def save_sparse_X(self, X):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

    def get_train_command(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

    def find_model_file(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

    def get_test_command(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_ERROR_DESC)

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
