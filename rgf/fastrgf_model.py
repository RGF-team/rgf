from __future__ import absolute_import

import os
import subprocess
from uuid import uuid4

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_X_y, column_or_1d

from rgf import util
import rgf


class FastRGFRegressor(util.RGFRegressorBase):
    """
    A Fast Regularized Greedy Forest regressor by Tong Zhang.
    See https://github.com/baidu/fast_rgf

    This function is alpha version.
    The part of the function may be not tested, not documented and not
    unstabled. API can be changed in the future.

    Parameters
    ----------
    dtree_new_tree_gain_ratio : float, optional (default=1.0)
        new tree is created when leaf-nodes gain < this value * estimated gain
        of creating new three.

    loss : string ("LS" or "MODLS" or "LOGISTIC"), optional (default="LS")

    dtree_lamL1 : float, optional (default=1.0) L1 regularization parameter.

    dtree_lamL2 : float, optional (default=1000.0) L2 regularization parameter.

    forest_ntrees : int, optional (default=500) number of trees.

    discretize_dense_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    discretize_dense_lamL2 : float, optional (default=2.0)
        L2 regularization parameter for discretization.

    discretize_sparse_max_features : int, optional (default=80000)
        maximum number of selected features.

    discretize_sparse_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    """
    # TODO(fukatani): Test
    # TODO(fukatani): Other parameter
    def __init__(self,
                 dtree_new_tree_gain_ratio=1.0,
                 dtree_loss="LS",
                 dtree_lamL1=1,
                 dtree_lamL2=1000,
                 forest_ntrees=500,
                 discretize_dense_max_buckets=65000,
                 discretize_dense_lamL2=10,
                 discretize_sparse_max_features=80000,
                 discretize_sparse_max_buckets=200,
                 n_iter=None,
                 verbose=0):
        if not rgf.fastrgf_available():
            raise Exception('FastRGF is not installed correctly.')
        self.dtree_new_tree_gain_ratio = dtree_new_tree_gain_ratio
        self.dtree_loss = dtree_loss
        self.dtree_lamL1 = dtree_lamL1
        self.dtree_lamL2 = dtree_lamL2
        self.forest_ntrees = forest_ntrees
        self.discretize_dense_max_buckets = discretize_dense_max_buckets
        self.discretize_dense_lamL2 = discretize_dense_lamL2
        self.discretize_sparse_max_features = discretize_sparse_max_features
        self.discretize_sparse_max_buckets = discretize_sparse_max_buckets

        self.n_iter = n_iter
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(rgf.COUNTER.increment())
        rgf.UUIDS.append(self._file_prefix)
        self._n_features = None
        self._fitted = None
        self._latest_model_loc = None
        self.model_file = None

    def fit(self, X, y, sample_weight=None):
        """
        Build a Fast RGF Regressor from the training set (X, y).

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
        # _validate_params(**self.get_params())

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=False, y_numeric=True)
        n_samples, self._n_features = X.shape

        if self.n_iter is None:
            if self.dtree_loss == "LS":
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

        train_x_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".train.data.x")
        train_y_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".train.data.weight")
        self.model_file = os.path.join(rgf.get_temp_path(), self._file_prefix + ".model")
        if sp.isspmatrix(X):
            util.sparse_savetxt(train_x_loc, X, including_header=False)
        else:
            np.savetxt(train_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        # Format train command
        cmd = []
        cmd.append(rgf.get_fastrgf_path() + "/forest_train")
        cmd.append("forest.ntrees=%s" % self.forest_ntrees)
        cmd.append("discretize.dense.lamL2=%s" % self.discretize_dense_lamL2)
        cmd.append("discretize.sparse.max_features=%s" % self.discretize_sparse_max_features)
        cmd.append("discretize.sparse.max_buckets=%s" % self.discretize_sparse_max_buckets)
        cmd.append("discretize.dense.max_buckets=%s" % self.discretize_dense_max_buckets)
        cmd.append("dtree.new_tree_gain_ratio=%s" % self.dtree_new_tree_gain_ratio)
        cmd.append("dtree.loss=%s" % self.dtree_loss)
        cmd.append("dtree.lamL1=%s" % self.dtree_lamL1)
        cmd.append("dtree.lamL2=%s" % self.dtree_lamL2)
        if sp.isspmatrix(X):
            cmd.append("trn.x-file_format=x.sparse")
        cmd.append("trn.x-file=%s" % train_x_loc)
        cmd.append("trn.y-file=%s" % train_y_loc)
        cmd.append("trn.w-file=%s" % train_weight_loc)
        cmd.append("trn.target=REAL")
        cmd.append("set.verbose=%s" % self.verbose)
        cmd.append("model.save=%s" % self.model_file)

        print(' '.join(cmd))

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        if not os.path.isfile(self.model_file):
            raise Exception("Training is abnormally finished.")

        self._fitted = True

        # Find latest model location
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
            raise NotFittedError(util.not_fitted_error_desc())

        X = check_array(X, accept_sparse=True)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))

        test_x_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".test.data.x")
        if sp.isspmatrix(X):
            util.sparse_savetxt(test_x_loc, X, including_header=False)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        if not os.path.isfile(self.model_file):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(rgf.get_temp_path()))

        # Format test command
        pred_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".predictions.txt")

        cmd = []
        cmd.append(rgf.get_fastrgf_path() + "/forest_predict")
        cmd.append("model.load=%s" % self.model_file)
        cmd.append("tst.x-file=%s" % test_x_loc)
        if sp.isspmatrix(X):
            cmd.append("tst.x-file_format=x.sparse")
        cmd.append("tst.target=REAL")
        cmd.append("tst.output-prediction=%s" % pred_loc)

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT).communicate()

        if self.verbose:
            for k in output:
                print(k)

        y_pred = np.loadtxt(pred_loc)
        return y_pred

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._fitted:
            with open(self.model_file, 'rb') as fr:
                state["model"] = fr.read()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._fitted:
            with open(self.model_file, 'wb') as fw:
                fw.write(self.__dict__["model"])
            del self.__dict__["model"]


class FastRGFClassifier(util.RGFClassifierBase, RegressorMixin):
    """
    A Fast Regularized Greedy Forest classifier by Tong Zhang.
    See https://github.com/baidu/fast_rgf

    This function is alpha version.
    The part of the function may be not tested, not documented and not
    unstabled. API can be changed in the future.

    Parameters
    ----------
    dtree_new_tree_gain_ratio : float, optional (default=1.0)
        new tree is created when leaf-nodes gain < this value * estimated gain
        of creating new three.

    loss : string ("LS" or "MODLS" or "LOGISTIC"), optional (default="LS")

    dtree_lamL1 : float, optional (default=1.0) L1 regularization parameter.

    dtree_lamL2 : float, optional (default=1000.0) L2 regularization parameter.

    forest_ntrees : int, optional (default=500) number of trees.

    discretize_dense_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    discretize_dense_lamL2 : float, optional (default=2.0)
        L2 regularization parameter for discretization.

    discretize_sparse_max_features : int, optional (default=80000)
        maximum number of selected features.

    discretize_sparse_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    """
    # TODO(fukatani): Test
    # TODO(fukatani): Other parameter
    def __init__(self,
                 dtree_new_tree_gain_ratio=1.0,
                 dtree_loss="LS",  # "MODLS" or "LOGISTIC" or "LS"
                 dtree_lamL1=10,
                 dtree_lamL2=1000,
                 forest_ntrees=1000,
                 discretize_dense_max_buckets=250,
                 discretize_dense_lamL2=10,
                 discretize_sparse_max_features=10,
                 discretize_sparse_max_buckets=10,
                 n_iter=None,
                 calc_prob="sigmoid",
                 verbose=0):
        self.dtree_new_tree_gain_ratio = dtree_new_tree_gain_ratio
        self.dtree_loss = dtree_loss
        self.dtree_lamL1 = dtree_lamL1
        self.dtree_lamL2 = dtree_lamL2
        self.forest_ntrees = forest_ntrees
        self.discretize_dense_max_buckets = discretize_dense_max_buckets
        self.discretize_dense_lamL2 = discretize_dense_lamL2
        self.discretize_sparse_max_features = discretize_sparse_max_features
        self.discretize_sparse_max_buckets = discretize_sparse_max_buckets

        self.n_iter = n_iter
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(rgf.COUNTER.increment())
        rgf.UUIDS.append(self._file_prefix)
        self._fitted = None
        self._latest_model_loc = None
        self.model_file = None
        self._estimators = None
        self._classes = None
        self._classes_map = {}
        self._n_classes = None
        self._n_features = None
        self._fitted = None
        self.calc_prob = calc_prob

    def fit(self, X, y, sample_weight=None):
        """
        Build a Fast RGF Classifier from the training set (X, y).

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

        X, y = check_X_y(X, y, accept_sparse=True)
        n_samples, self._n_features = X.shape

        if self.n_iter is None:
            if self.dtree_loss == "LS":
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

        params = dict(dtree_new_tree_gain_ratio=self.dtree_new_tree_gain_ratio,
                      dtree_loss=self.dtree_loss,  # "MODLS" or "LOGISTIC" or "LS"
                      dtree_lamL1=self.dtree_lamL1,
                      dtree_lamL2=self.dtree_lamL2,
                      forest_ntrees=self.forest_ntrees,
                      discretize_dense_max_buckets=self.discretize_dense_max_buckets,
                      discretize_dense_lamL2=self.discretize_dense_lamL2,
                      discretize_sparse_max_features=self.discretize_sparse_max_features,
                      discretize_sparse_max_buckets=self.discretize_sparse_max_buckets,
                      n_iter=self.n_iter,
                      verbose=self.verbose)

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            if (sample_weight <= 0).any():
                raise ValueError("Sample weights must be positive.")

        if self._n_classes == 2:
            self._classes_map[0] = self._classes[0]
            self._classes_map[1] = self._classes[1]
            self._estimators = [None]
            y = (y == self._classes[0]).astype(int)
            self._estimators[0] = _FastRGFBinaryClassifier(**params)
            self._estimators[0].fit(X, y, sample_weight)
        elif self._n_classes > 2:
            if sp.isspmatrix_dok(X):
                X = X.tocsr().tocoo()  # Fix to avoid scipy 7699 issue
            self._estimators = [None] * self._n_classes
            ovr_list = [None] * self._n_classes
            for i, cls_num in enumerate(self._classes):
                self._classes_map[i] = cls_num
                ovr_list[i] = (y == cls_num).astype(int)
                self._estimators[i] = _FastRGFBinaryClassifier(**params)

            self._estimators = [self._estimators[i].fit(X, ovr_list[i], sample_weight)
                                for i in range(self._n_classes)]
        else:
            raise ValueError("Classifier can't predict when only one class is present.")

        self._fitted = True
        return self


class _FastRGFBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Fast RGF Binary Classifier.
    Don't instantiate this class directly.
    This class should be instantiated only by FastRGFClassifier.
    """
    def __init__(self,
                 dtree_new_tree_gain_ratio=1.0,
                 dtree_loss="LS",  # "MODLS" or "LOGISTIC" or "LS"
                 dtree_lamL1=10,
                 dtree_lamL2=1000,
                 forest_ntrees=1000,
                 discretize_dense_max_buckets=250,
                 discretize_dense_lamL2=10,
                 discretize_sparse_max_features=10,
                 discretize_sparse_max_buckets=10,
                 n_iter=None,
                 verbose=0):
        self.dtree_new_tree_gain_ratio = dtree_new_tree_gain_ratio
        self.dtree_loss = dtree_loss
        self.dtree_lamL1 = dtree_lamL1
        self.dtree_lamL2 = dtree_lamL2
        self.forest_ntrees = forest_ntrees
        self.discretize_dense_max_buckets = discretize_dense_max_buckets
        self.discretize_dense_lamL2 = discretize_dense_lamL2
        self.discretize_sparse_max_features = discretize_sparse_max_features
        self.discretize_sparse_max_buckets = discretize_sparse_max_buckets

        self.n_iter = n_iter
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(rgf.COUNTER.increment())
        rgf.UUIDS.append(self._file_prefix)
        self._fitted = None
        self.model_file = None
        self._estimators = None
        self._classes = None
        self._n_classes = None
        self._n_features = None

    def fit(self, X, y, sample_weight):
        train_x_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".train.data.x")
        train_y_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".train.data.weight")
        self.model_file = os.path.join(rgf.get_temp_path(), self._file_prefix + ".model")
        if sp.isspmatrix(X):
            util.sparse_savetxt(train_x_loc, X, including_header=False)
        else:
            np.savetxt(train_x_loc, X, delimiter=' ', fmt="%s")

        # Convert 1 to 1, 0 to -1
        y = 2 * y - 1
        np.savetxt(train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        # Format train command

        cmd = []
        cmd.append(rgf.get_fastrgf_path() + "/forest_train")
        cmd.append("forest.ntrees=%s" % self.forest_ntrees)
        cmd.append("discretize.dense.lamL2=%s" % self.discretize_dense_lamL2)
        cmd.append("discretize.sparse.max_features=%s" % self.discretize_sparse_max_features)
        cmd.append("discretize.sparse.max_buckets=%s" % self.discretize_sparse_max_buckets)
        cmd.append("discretize.dense.max_buckets=%s" % self.discretize_dense_max_buckets)
        cmd.append("dtree.new_tree_gain_ratio=%s" % self.dtree_new_tree_gain_ratio)
        cmd.append("dtree.loss=%s" % self.dtree_loss)
        cmd.append("dtree.lamL1=%s" % self.dtree_lamL1)
        cmd.append("dtree.lamL2=%s" % self.dtree_lamL2)
        cmd.append("trn.x-file=%s" % train_x_loc)
        cmd.append("trn.y-file=%s" % train_y_loc)
        cmd.append("trn.w-file=%s" % train_weight_loc)
        if sp.isspmatrix(X):
            cmd.append("trn.x-file_format=x.sparse")
        cmd.append("trn.target=BINARY")
        cmd.append("set.verbose=%s" % self.verbose)
        cmd.append("model.save=%s" % self.model_file)

        # Train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        self._fitted = True
        if not self.model_file:
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(rgf.get_temp_path()))
        return self

    def predict_proba(self, X):
        if self._fitted is None:
            raise NotFittedError(util.not_fitted_error_desc())
        if not os.path.isfile(self.model_file):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(rgf.get_temp_path()))

        test_x_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".test.data.x")
        if sp.isspmatrix(X):
            util.sparse_savetxt(test_x_loc, X, including_header=False)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        # Format test command
        pred_loc = os.path.join(rgf.get_temp_path(), self._file_prefix + ".predictions.txt")

        cmd = []
        cmd.append(rgf.get_fastrgf_path() + "/forest_predict")
        cmd.append("model.load=%s" % self.model_file)
        cmd.append("tst.x-file=%s" % test_x_loc)
        if sp.isspmatrix(X):
            cmd.append("tst.x-file_format=x.sparse")
        cmd.append("tst.target=REAL")
        cmd.append("tst.output-prediction=%s" % pred_loc)

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT).communicate()

        if self.verbose:
            for k in output:
                print(k)

        return np.loadtxt(pred_loc)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._fitted:
            with open(self.model_file, 'rb') as fr:
                state["model"] = fr.read()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._fitted:
            with open(self.model_file, 'wb') as fw:
                fw.write(self.__dict__["model"])
            del self.__dict__["model"]
