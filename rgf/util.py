from __future__ import absolute_import


import atexit
import codecs
import glob
import os
import platform
import stat
import subprocess
from threading import Lock

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
                n_removed_files += cleanup_partial(est._file_prefix,
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
