import atexit
import codecs
import glob
import numbers
import os
import platform
import stat
import subprocess
import warnings
from configparser import MissingSectionHeaderError, RawConfigParser
from io import StringIO
from tempfile import gettempdir
from threading import Lock
from uuid import uuid4

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_X_y, column_or_1d


FLOATS = (float, np.float16, np.float32, np.float64, np.double)
INTS = (numbers.Integral, np.integer)
NOT_FITTED_ERROR_DESC = "Estimator not fitted, call `fit` before exploiting the model."
NOT_IMPLEMENTED_ERROR_DESC = "This method isn't implemented in base class."
SYSTEM = platform.system()
UUIDS = []


@atexit.register
def cleanup():
    for uuid in UUIDS:
        cleanup_partial(uuid)


def cleanup_partial(uuid, remove_from_list=False):
    n_removed_files = 0
    if uuid in UUIDS:
        model_glob = os.path.join(Config().TEMP_PATH, uuid + "*")
        for fn in glob.glob(model_glob):
            os.remove(fn)
            n_removed_files += 1
        if remove_from_list:
            UUIDS.remove(uuid)
    return n_removed_files


class Config(object):
    DEFAULT_RGF_PATH = None
    RGF_PATH = None
    DEFAULT_FASTRGF_PATH = None
    FASTRGF_PATH = None
    TEMP_PATH = None
    CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

    (FASTRGF_TRAIN_EXECUTABLE_FILE,
     FASTRGF_PREDICT_EXECUTABLE_FILE) = (("forest_train.exe", "forest_predict.exe")
                                         if SYSTEM in ('Windows', 'Microsoft')
                                         else ("forest_train", "forest_predict"))

    RGF_AVAILABLE = None
    FASTRGF_AVAILABLE = None

    def __init__(self, verbose=False):
        (self.DEFAULT_RGF_PATH, self.RGF_PATH,
         self.DEFAULT_FASTRGF_PATH, self.FASTRGF_PATH, self.TEMP_PATH) = self.init_paths()
        (self.RGF_PATH, self.RGF_AVAILABLE,
         self.FASTRGF_PATH, self.FASTRGF_AVAILABLE) = self.set_paths_and_availability(verbose)

    @classmethod
    def init_paths(cls):
        if cls.TEMP_PATH is None:
            config = RawConfigParser()
            path = os.path.join(os.path.expanduser('~'), '.rgfrc')

            try:
                with codecs.open(path, 'r', 'utf-8') as cfg:
                    with StringIO(cfg.read()) as strIO:
                        config.read_file(strIO)
            except MissingSectionHeaderError:
                with codecs.open(path, 'r', 'utf-8') as cfg:
                    with StringIO('[glob]\n' + cfg.read()) as strIO:
                        config.read_file(strIO)
            except Exception:
                pass

            if SYSTEM in ('Windows', 'Microsoft'):
                try:
                    cls.RGF_PATH = os.path.abspath(config.get(config.sections()[0], 'rgf_location'))
                except Exception:
                    cls.RGF_PATH = os.path.join(os.path.expanduser('~'), 'rgf.exe')
                cls.DEFAULT_RGF_PATH = 'rgf.exe'
            else:  # Linux, Darwin (macOS), etc.
                try:
                    cls.RGF_PATH = os.path.abspath(config.get(config.sections()[0], 'rgf_location'))
                except Exception:
                    cls.RGF_PATH = os.path.join(os.path.expanduser('~'), 'rgf')
                cls.DEFAULT_RGF_PATH = 'rgf'

            try:
                cls.FASTRGF_PATH = os.path.abspath(config.get(config.sections()[0], 'fastrgf_location'))
            except Exception:
                cls.FASTRGF_PATH = os.path.expanduser('~')
            cls.DEFAULT_FASTRGF_PATH = ''

            try:
                cls.TEMP_PATH = os.path.abspath(config.get(config.sections()[0], 'temp_location'))
            except Exception:
                cls.TEMP_PATH = os.path.join(gettempdir(), 'rgf')
            if not os.path.isdir(cls.TEMP_PATH):
                os.makedirs(cls.TEMP_PATH)
            if not os.access(cls.TEMP_PATH, os.W_OK):
                raise Exception("{0} is not writable directory. Please set "
                                "config flag 'temp_location' to writable directory".format(cls.TEMP_PATH))

        return cls.DEFAULT_RGF_PATH, cls.RGF_PATH, cls.DEFAULT_FASTRGF_PATH, cls.FASTRGF_PATH, cls.TEMP_PATH

    @classmethod
    def is_rgf_executable(cls, path):
        temp_x_loc = os.path.join(cls.TEMP_PATH, 'temp_rgf.train.data.x')
        temp_y_loc = os.path.join(cls.TEMP_PATH, 'temp_rgf.train.data.y')
        temp_model_loc = os.path.join(cls.TEMP_PATH, 'temp_rgf.model')
        temp_pred_loc = os.path.join(cls.TEMP_PATH, 'temp_rgf.predictions.txt')
        np.savetxt(temp_x_loc, [[1, 0, 1, 0], [0, 1, 0, 1]], delimiter=' ', fmt="%s")
        np.savetxt(temp_y_loc, [1, -1], delimiter=' ', fmt="%s")
        UUIDS.append('temp_rgf')
        params_train = []
        params_train.append("train_x_fn=%s" % temp_x_loc)
        params_train.append("train_y_fn=%s" % temp_y_loc)
        params_train.append("model_fn_prefix=%s" % temp_model_loc)
        params_train.append("reg_L2=%s" % 1)
        params_train.append("max_leaf_forest=%s" % 10)
        params_pred = []
        params_pred.append("test_x_fn=%s" % temp_x_loc)
        params_pred.append("prediction_fn=%s" % temp_pred_loc)
        params_pred.append("model_fn=%s" % temp_model_loc + "-01")
        try:
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
        try:
            subprocess.check_output((path, "train", ",".join(params_train)),
                                    stderr=subprocess.STDOUT)
            subprocess.check_output((path, "predict", ",".join(params_pred)),
                                    stderr=subprocess.STDOUT)
        except Exception:
            return False
        return True

    @classmethod
    def is_fastrgf_executable(cls, path, verbose=False):
        if verbose:
            print("[FastRGF] Investigate {}".format(path))
        temp_x_loc = os.path.join(cls.TEMP_PATH, 'temp_fastrgf.train.data.x')
        temp_y_loc = os.path.join(cls.TEMP_PATH, 'temp_fastrgf.train.data.y')
        temp_model_loc = os.path.join(cls.TEMP_PATH, "temp_fastrgf.model")
        temp_pred_loc = os.path.join(cls.TEMP_PATH, "temp_fastrgf.predictions.txt")
        X = np.tile(np.array([[1, 0, 1, 0], [0, 1, 0, 1]]), (14, 1))
        y = np.tile(np.array([1, -1]), 14)
        np.savetxt(temp_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(temp_y_loc, y, delimiter=' ', fmt="%s")
        UUIDS.append('temp_fastrgf')
        path_train = os.path.join(path, "forest_train")
        params_train = []
        params_train.append("forest.ntrees=%s" % 10)
        params_train.append("tst.target=%s" % "BINARY")
        params_train.append("trn.x-file=%s" % temp_x_loc)
        params_train.append("trn.y-file=%s" % temp_y_loc)
        params_train.append("model.save=%s" % temp_model_loc)
        cmd_train = [path_train]
        cmd_train.extend(params_train)
        path_pred = os.path.join(path, "forest_predict")
        params_pred = []
        params_pred.append("model.load=%s" % temp_model_loc)
        params_pred.append("tst.x-file=%s" % temp_x_loc)
        params_pred.append("tst.output-prediction=%s" % temp_pred_loc)
        cmd_pred = [path_pred]
        cmd_pred.extend(params_pred)
        try:
            os.chmod(path_train, os.stat(path_train).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.chmod(path_pred, os.stat(path_pred).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
        try:
            subprocess.check_output(cmd_train, stderr=subprocess.STDOUT)
            subprocess.check_output(cmd_pred, stderr=subprocess.STDOUT)
        except Exception as e:
            if verbose:
                print(e)
            return False

        if verbose:
            print("[FastRGF] Found in {}".format(path))

        return True

    @classmethod
    def set_paths_and_availability(cls, verbose=False):
        if cls.RGF_AVAILABLE is None or cls.FASTRGF_AVAILABLE is None:
            cls.RGF_AVAILABLE = True
            if cls.is_rgf_executable(os.path.join(cls.CURRENT_DIR, cls.DEFAULT_RGF_PATH)):
                cls.RGF_PATH = os.path.join(cls.CURRENT_DIR, cls.DEFAULT_RGF_PATH)
            elif cls.is_rgf_executable(cls.DEFAULT_RGF_PATH):
                cls.RGF_PATH = cls.DEFAULT_RGF_PATH
            elif cls.is_rgf_executable(cls.RGF_PATH):
                pass
            else:
                cls.RGF_AVAILABLE = False
                warnings.warn("Cannot find RGF executable file. "
                              "RGF estimators will be unavailable for usage.")

            cls.FASTRGF_AVAILABLE = True
            if cls.is_fastrgf_executable(cls.CURRENT_DIR, verbose):
                cls.FASTRGF_PATH = cls.CURRENT_DIR
            elif cls.is_fastrgf_executable(cls.DEFAULT_FASTRGF_PATH, verbose):
                cls.FASTRGF_PATH = cls.DEFAULT_FASTRGF_PATH
            elif cls.is_fastrgf_executable(cls.FASTRGF_PATH, verbose):
                pass
            else:
                cls.FASTRGF_AVAILABLE = False
                warnings.warn("Cannot find FastRGF executable files. "
                              "FastRGF estimators will be unavailable for usage.")

        return cls.RGF_PATH, cls.RGF_AVAILABLE, cls.FASTRGF_PATH, cls.FASTRGF_AVAILABLE


class AtomicCounter(object):
    def __init__(self):
        self.value = 0
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value


CONFIG = None
COUNTER = AtomicCounter()


def sparse_savetxt(filename, input_array, including_header=True):
    if sp.isspmatrix_csr(input_array):
        input_array = input_array.tocoo()
    else:
        input_array = input_array.tocsr().tocoo()
    n_row = input_array.shape[0]
    current_sample_row = 0
    line = []
    with open(filename, 'w') as fw:
        if including_header:
            fw.write('sparse {0:d}\n'.format(input_array.shape[-1]))
        for i, j, v in zip(input_array.row, input_array.col, input_array.data):
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


def fit_ovr_binary(binary_clf, X, y, sample_weight):
    return binary_clf.fit(X, y, sample_weight)


class CommonRGFExecuterBase(BaseEstimator):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._file_prefix = str(uuid4()) + str(COUNTER.increment())
        UUIDS.append(self._file_prefix)
        self._fitted = False

    def _set_paths(self):
        self._train_x_loc = os.path.join(self.config.TEMP_PATH, self._file_prefix + ".train.data.x")
        self._test_x_loc = os.path.join(self.config.TEMP_PATH, self._file_prefix + ".test.data.x")
        self._train_y_loc = os.path.join(self.config.TEMP_PATH, self._file_prefix + ".train.data.y")
        self._train_weight_loc = os.path.join(self.config.TEMP_PATH, self._file_prefix + ".train.data.weight")
        self._model_file_loc = os.path.join(self.config.TEMP_PATH, self._file_prefix + ".model")
        self._pred_loc = os.path.join(self.config.TEMP_PATH, self._file_prefix + ".predictions.txt")
        self._feature_importances_loc = os.path.join(self.config.TEMP_PATH, self._file_prefix + ".feature_importances.txt")

    def _execute_command(self, cmd, force_verbose=False):
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True)
        output = process.communicate()
        output = '\n'.join([i for i in output if i is not None])

        if process.returncode != 0:
            raise Exception(output)
        elif self.verbose or force_verbose:
            print(output)

    def _save_test_X(self, X):
        if sp.isspmatrix(X):
            self._save_sparse_X(self._test_x_loc, X)
            is_sparse_test_X = True
        else:
            np.savetxt(self._test_x_loc, X, delimiter=' ', fmt="%s")
            is_sparse_test_X = False

        return is_sparse_test_X

    def _save_train_data(self, X, y, sample_weight):
        if sample_weight is None:
            self._use_sample_weight = False
        else:
            self._use_sample_weight = True

        if sp.isspmatrix(X):
            self._save_sparse_X(self._train_x_loc, X)
            np.savetxt(self._train_y_loc, y, delimiter=' ', fmt="%s")
            if self._use_sample_weight:
                np.savetxt(self._train_weight_loc, sample_weight, delimiter=' ', fmt="%s")
            self._is_sparse_train_X = True
        else:
            self._save_dense_files(X, y, sample_weight)
            self._is_sparse_train_X = False

    def fit(self, X, y, sample_weight):
        self._set_paths()
        if self.is_classification:
            # Convert 1 to 1, 0 to -1
            y = 2 * y - 1
        self._save_train_data(X, y, sample_weight)
        cmd = self._get_train_command()
        self._execute_command(cmd)

        self._find_model_file()
        self._fitted = True
        return self

    def predict(self, X):
        self._check_fitted()

        is_sparse_test_X = self._save_test_X(X)

        cmd = self._get_test_command(is_sparse_test_X)
        self._execute_command(cmd)

        return np.loadtxt(self._pred_loc)

    def _check_fitted(self):
        if not self._fitted:
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        if not os.path.isfile(self._model_file):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(self.config.TEMP_PATH))

    def _save_sparse_X(self, path, X):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def _save_dense_files(self, X, y, sample_weight):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def _find_model_file(self):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def _get_train_command(self):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def _get_test_command(self, is_sparse_test_X):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, "_model_file") and self._fitted:
            with open(self._model_file, 'rb') as fr:
                state["model"] = fr.read()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_model_file") and self._fitted:
            with open(self._model_file, 'wb') as fw:
                fw.write(self.__dict__["model"])
            del self.__dict__["model"]


class CommonRGFEstimatorBase(BaseEstimator):
    def __init__(self):
        self._set_config()

    @property
    def n_features_(self):
        """The number of features when `fit` is performed."""
        if not hasattr(self, '_n_features'):
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        else:
            return self._n_features

    @property
    def fitted_(self):
        """Indicates whether `fit` is performed."""
        if not hasattr(self, '_fitted') or not self._fitted:
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        else:
            return True

    def _get_sample_weight(self, sample_weight):
        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight, warn=True)
            if (sample_weight <= 0).any():
                raise ValueError("Sample weights must be positive.")
        return sample_weight

    def _check_n_features(self, n_features):
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))

    def _validate_params(self, params):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def _set_params_with_dependencies(self):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    @property
    def estimators_(self):
        """The collection of fitted sub-estimators when `fit` is performed."""
        if not hasattr(self, '_estimators'):
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        else:
            return self._estimators

    def cleanup(self):
        """
        Remove tempfiles used by this model.
        Note, that after calling this method you'll need to refit the estimator.

        Returns
        -------
        n_removed_files : int
            Returns the number of removed files.
        """
        n_removed_files = 0
        if hasattr(self, '_estimators'):
            for est in self._estimators:
                n_removed_files += cleanup_partial(est._file_prefix,
                                                   remove_from_list=True)

        # No more able to predict without refitting.
        self._fitted = False
        return n_removed_files

    def _get_params(self):
        return {'config': CONFIG}

    @classmethod
    def _set_config(cls):
        global CONFIG
        if CONFIG is None:
            CONFIG = Config()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["config"] = CONFIG
        return state

    def __setstate__(self, state):
        global CONFIG
        CONFIG = state.pop("config")
        self.__dict__.update(state)

    def _fit_binary_task(self, X, y, sample_weight, params):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def _fit_multiclass_task(self, X, y, sample_weight, params):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)

    def _fit_regression_task(self, X, y, sample_weight, params):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_DESC)


class RGFClassifierMixin(object):
    def _more_tags(self):
        return {'X_types': ['2darray', 'sparse', '1dlabels'],
                'requires_y' : True,
                '_xfail_checks': {'check_sample_weights_invariance': 'sample_weight must have positive values'}}

    @property
    def classes_(self):
        """The classes labels when `fit` is performed."""
        if not hasattr(self, '_classes'):
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        else:
            return self._classes

    @property
    def n_classes_(self):
        """The number of classes when `fit` is performed."""
        if not hasattr(self, '_n_classes'):
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        else:
            return self._n_classes

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
        self._validate_params(**self.get_params())

        X, y = check_X_y(X, y, accept_sparse=True)
        if sp.isspmatrix(X):
            self._is_sparse_train_X = True
        else:
            self._is_sparse_train_X = False
        self._n_samples, self._n_features = X.shape
        sample_weight = self._get_sample_weight(sample_weight)
        check_consistent_length(X, y, sample_weight)
        check_classification_targets(y)
        self._classes = sorted(np.unique(y))
        self._n_classes = len(self._classes)
        self._classes_map = {}

        self._set_params_with_dependencies()
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
        self.n_features_in_ = self._n_features

        return self

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
        if not hasattr(self, '_fitted') or not self._fitted:
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        X = check_array(X, accept_sparse=True)
        self._check_n_features(X.shape[1])

        if self._n_classes == 2:
            y = self._estimators[0].predict(X)
            y = sigmoid(y)
            y = np.c_[y, 1 - y]
        else:
            y = np.zeros((X.shape[0], self._n_classes))
            for i, clf in enumerate(self._estimators):
                class_proba = clf.predict(X)
                y[:, i] = class_proba

            if self.calc_prob == "sigmoid":
                y = sigmoid(y)
                normalizer = np.sum(y, axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                y /= normalizer
            else:
                y = softmax(y)
        return y


class RGFRegressorMixin(object):
    def _more_tags(self):
        return {'X_types': ['2darray', 'sparse', '1dlabels'],
                'requires_y' : True,
                '_xfail_checks': {'check_sample_weights_invariance': 'sample_weight must have positive values'}}

    def fit(self, X, y, sample_weight=None):
        """
        Build a regressor from the training set (X, y).

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
        self._validate_params(**self.get_params())

        X, y = check_X_y(X, y, accept_sparse=True)
        if sp.isspmatrix(X):
            self._is_sparse_train_X = True
        else:
            self._is_sparse_train_X = False
        self._n_samples, self._n_features = X.shape
        sample_weight = self._get_sample_weight(sample_weight)
        check_consistent_length(X, y, sample_weight)

        self._set_params_with_dependencies()
        params = self._get_params()

        self._estimators = [None]
        self._fit_regression_task(X, y, sample_weight, params)

        self._fitted = True
        self.n_features_in_ = self._n_features

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
        if not hasattr(self, '_fitted') or not self._fitted:
            raise NotFittedError(NOT_FITTED_ERROR_DESC)
        X = check_array(X, accept_sparse=True)
        self._check_n_features(X.shape[1])
        return self._estimators[0].predict(X)
