from __future__ import absolute_import

import os
import subprocess
from uuid import uuid4

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.externals.joblib import cpu_count
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_X_y, column_or_1d

from rgf import utils


def _validate_fast_rgf_params(**kwargs):
    pass


class FastRGFRegressor(utils.RGFRegressorBase):
    """
    A Fast Regularized Greedy Forest regressor by Tong Zhang.
    See https://github.com/baidu/fast_rgf

    This function is alpha version.
    The part of the function may be not tested, not documented and not
    unstabled. API can be changed in the future.

    Parameters
    ----------
    dtree_max_level : int, optional (default=6)
        maximum level of the tree.

    dtree_max_nodes : int, optional (default=50)
        maximum number of leaf nodes in best-first search.

    dtree_new_tree_gain_ratio : float, optional (default=1.0)
        new tree is created when leaf-nodes gain < this value * estimated gain
        of creating new three.

    dtree_min_sample : int, optional (default=5)
        minimum sample per node.

    loss : string ("LS" or "MODLS" or "LOGISTIC"), optional (default="LS")

    dtree_lamL1 : float, optional (default=1.0) L1 regularization parameter.

    dtree_lamL2 : float, optional (default=1000.0) L2 regularization parameter.

    forest_opt : 'rgf' or 'epsilon-greedy', optional (default='rgf')
        optimization method for training forest

    forest_ntrees : int, optional (default=500) number of trees.

    forest_stepsize : float optional (default=0.001)
        step size of epsilon-greedy boosting (inactive for rgf)

    discretize_dense_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    discretize_dense_lamL2 : float, optional (default=2.0)
        L2 regularization parameter for discretization.

    discretize_dense_min_bucket_weights : float, optional (default=5.0)
        minimum sum of data weights for each discretized value.

    discretize_sparse_max_features : int, optional (default=80000)
        maximum number of selected features.

    discretize_sparse_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    discretize_sparse_lamL2 : float, optional (default=2.0)
        L2 regularization parameter for discretization.

    discretize_sparse_min_bucket_weights : float, optional (default=5.0)
        minimum sum of data weights for each discretized value.

    discretize_sparse_min_occurences : int, optional (default=5)
        minimum number of occurrences for a feature to be selected

    """
    # TODO(fukatani): Test
    def __init__(self,
                 dtree_max_level=6,
                 dtree_max_nodes=50,
                 dtree_new_tree_gain_ratio=1.0,
                 dtree_min_sample=5,
                 dtree_loss="LS",
                 dtree_lamL1=1,
                 dtree_lamL2=1000,
                 forest_opt='rgf',
                 forest_ntrees=500,
                 forest_stepsize=0.001,
                 discretize_dense_max_buckets=65000,
                 discretize_dense_lamL2=2.0,
                 discretize_dense_min_bucket_weights=5.0,
                 discretize_sparse_max_features=80000,
                 discretize_sparse_max_buckets=200,
                 discretize_sparse_lamL2=2.0,
                 discretize_sparse_min_bucket_weights=5.0,
                 discretize_sparse_min_occurences=5,
                 n_iter=None,
                 n_jobs=-1,
                 verbose=0):
        if not utils.fastrgf_available():
            raise Exception('FastRGF is not installed correctly.')
        self.dtree_max_level = dtree_max_level
        self.dtree_max_nodes = dtree_max_nodes
        self.dtree_min_sample = dtree_min_sample
        self.dtree_new_tree_gain_ratio = dtree_new_tree_gain_ratio
        self.dtree_loss = dtree_loss
        self.dtree_lamL1 = dtree_lamL1
        self.dtree_lamL2 = dtree_lamL2
        self.forest_opt = forest_opt
        self.forest_ntrees = forest_ntrees
        self.forest_stepsize = forest_stepsize
        self.discretize_dense_max_buckets = discretize_dense_max_buckets
        self.discretize_dense_lamL2 = discretize_dense_lamL2
        self.discretize_dense_min_bucket_weights = discretize_dense_min_bucket_weights
        self.discretize_sparse_max_features = discretize_sparse_max_features
        self.discretize_sparse_max_buckets = discretize_sparse_max_buckets
        self.discretize_sparse_lamL2 = discretize_sparse_lamL2
        self.discretize_sparse_min_bucket_weights = discretize_sparse_min_bucket_weights
        self.discretize_sparse_min_occurences = discretize_sparse_min_occurences

        self.n_iter = n_iter
        self.n_jobs = n_jobs
        if self.n_jobs == -1:
            self.nthreads = 0
        elif n_jobs < 0:
            self.nthreads = cpu_count() + n_jobs + 1
        else:
            self.nthreads = n_jobs
        self.verbose = verbose
        self._file_prefix = str(uuid4()) + str(utils.COUNTER.increment())
        utils.UUIDS.append(self._file_prefix)
        self._n_features = None
        self._fitted = None
        self._latest_model_loc = None
        self.model_file = None

    def fit(self, X, y, sample_weight=None):
        """
        Build a FastRGF Regressor from the training set (X, y).

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

        train_x_loc = os.path.join(utils.get_temp_path(), self._file_prefix + ".train.data.x")
        train_y_loc = os.path.join(utils.get_temp_path(), self._file_prefix + ".train.data.y")
        train_weight_loc = os.path.join(utils.get_temp_path(), self._file_prefix + ".train.data.weight")
        self.model_file = os.path.join(utils.get_temp_path(), self._file_prefix + ".model")
        if sp.isspmatrix(X):
            utils.sparse_savetxt(train_x_loc, X, including_header=False)
        else:
            np.savetxt(train_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

        # Format train command
        cmd = []
        cmd.append(utils.get_fastrgf_path() + "/forest_train")
        cmd.append("forest.ntrees=%s" % self.forest_ntrees)
        cmd.append("forest.stepsize=%s" % self.forest_stepsize)
        cmd.append("forest.opt=%s" % self.forest_opt)
        cmd.append("discretize.dense.max_buckets=%s" % self.discretize_dense_max_buckets)
        cmd.append("discretize.dense.lamL2=%s" % self.discretize_dense_lamL2)
        cmd.append("discretize.dense.min_bucket_weights=%s" % self.discretize_dense_min_bucket_weights)
        cmd.append("discretize.sparse.max_features=%s" % self.discretize_sparse_max_features)
        cmd.append("discretize.sparse.max_buckets=%s" % self.discretize_sparse_max_buckets)
        cmd.append("discretize.sparse.lamL2=%s" % self.discretize_sparse_lamL2)
        cmd.append("discretize.sparse.min_bucket_weights=%s" % self.discretize_sparse_min_bucket_weights)
        cmd.append("discretize.sparse.min_occrrences=%s" % self.discretize_sparse_min_occurences)
        cmd.append("dtree.max_level=%s" % self.dtree_max_level)
        cmd.append("dtree.max_nodes=%s" % self.dtree_max_nodes)
        cmd.append("dtree.new_tree_gain_ratio=%s" % self.dtree_new_tree_gain_ratio)
        cmd.append("dtree.min_sample=%s" % self.dtree_min_sample)
        cmd.append("dtree.loss=%s" % self.dtree_loss)
        cmd.append("dtree.lamL1=%s" % self.dtree_lamL1)
        cmd.append("dtree.lamL2=%s" % self.dtree_lamL2)
        if sp.isspmatrix(X):
            cmd.append("trn.x-file_format=x.sparse")
        cmd.append("trn.x-file=%s" % train_x_loc)
        cmd.append("trn.y-file=%s" % train_y_loc)
        cmd.append("trn.w-file=%s" % train_weight_loc)
        cmd.append("trn.target=REAL")
        cmd.append("set.nthreads=%s" % self.nthreads)
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
            raise NotFittedError(utils.not_fitted_error_desc())

        X = check_array(X, accept_sparse=True)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))

        test_x_loc = os.path.join(utils.get_temp_path(), self._file_prefix + ".test.data.x")
        if sp.isspmatrix(X):
            utils.sparse_savetxt(test_x_loc, X, including_header=False)
        else:
            np.savetxt(test_x_loc, X, delimiter=' ', fmt="%s")

        if not os.path.isfile(self.model_file):
            raise Exception('Model learning result is not found in {0}. '
                            'This is rgf_python error.'.format(utils.get_temp_path()))

        # Format test command
        pred_loc = os.path.join(utils.get_temp_path(), self._file_prefix + ".predictions.txt")

        cmd = []
        cmd.append(utils.get_fastrgf_path() + "/forest_predict")
        cmd.append("model.load=%s" % self.model_file)
        cmd.append("tst.x-file=%s" % test_x_loc)
        if sp.isspmatrix(X):
            cmd.append("tst.x-file_format=x.sparse")
        cmd.append("tst.target=REAL")
        cmd.append("tst.output-prediction=%s" % pred_loc)
        cmd.append("set.nthreads=%s" % self.nthreads)
        cmd.append("set.verbose=%s" % self.verbose)

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


class FastRGFClassifier(utils.RGFClassifierBase):
    """
    A Fast Regularized Greedy Forest classifier by Tong Zhang.
    See https://github.com/baidu/fast_rgf

    This function is alpha version.
    The part of the function may be not tested, not documented and not
    unstabled. API can be changed in the future.

    Parameters
    ----------
    dtree_max_level : int, optional (default=6)
        maximum level of the tree.

    dtree_max_nodes : int, optional (default=50)
        maximum number of leaf nodes in best-first search.

    dtree_new_tree_gain_ratio : float, optional (default=1.0)
        new tree is created when leaf-nodes gain < this value * estimated gain
        of creating new three.

    dtree_min_sample : int, optional (default=5)
        minimum sample per node.

    loss : string ("LS" or "MODLS" or "LOGISTIC"), optional (default="LS")

    dtree_lamL1 : float, optional (default=1.0) L1 regularization parameter.

    dtree_lamL2 : float, optional (default=1000.0) L2 regularization parameter.

    forest_opt : 'rgf' or 'epsilon-greedy', optional (default='rgf')
        optimization method for training forest

    forest_ntrees : int, optional (default=500) number of trees.

    forest_stepsize : float optional (default=0.001)
        step size of epsilon-greedy boosting (inactive for rgf)

    discretize_dense_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    discretize_dense_lamL2 : float, optional (default=2.0)
        L2 regularization parameter for discretization.

    discretize_dense_min_bucket_weights : float, optional (default=5.0)
        minimum sum of data weights for each discretized value.

    discretize_sparse_max_features : int, optional (default=80000)
        maximum number of selected features.

    discretize_sparse_max_buckets : int, optional (default=200)
        maximum number of discretized values.

    discretize_sparse_lamL2 : float, optional (default=2.0)
        L2 regularization parameter for discretization.

    discretize_sparse_min_bucket_weights : float, optional (default=5.0)
        minimum sum of data weights for each discretized value.

    discretize_sparse_min_occurences : int, optional (default=5)
        minimum number of occurrences for a feature to be selected

    """
    # TODO(fukatani): Test
    def __init__(self,
                 dtree_max_level=6,
                 dtree_max_nodes=50,
                 dtree_new_tree_gain_ratio=1.0,
                 dtree_min_sample=5,
                 dtree_loss="LS",  # "MODLS" or "LOGISTIC" or "LS"
                 dtree_lamL1=10,
                 dtree_lamL2=1000,
                 forest_opt='rgf',
                 forest_ntrees=500,
                 forest_stepsize=0.001,
                 discretize_dense_max_buckets=250,
                 discretize_dense_lamL2=10,
                 discretize_dense_min_bucket_weights=5.0,
                 discretize_sparse_max_features=10,
                 discretize_sparse_max_buckets=10,
                 discretize_sparse_lamL2=2.0,
                 discretize_sparse_min_bucket_weights=5.0,
                 discretize_sparse_min_occurences=5,
                 n_iter=None,
                 calc_prob="sigmoid",
                 n_jobs=-1,
                 verbose=0):
        self.dtree_max_level = dtree_max_level
        self.dtree_max_nodes = dtree_max_nodes
        self.dtree_new_tree_gain_ratio = dtree_new_tree_gain_ratio
        self.dtree_min_sample = dtree_min_sample
        self.dtree_loss = dtree_loss
        self.dtree_lamL1 = dtree_lamL1
        self.dtree_lamL2 = dtree_lamL2
        self.forest_opt = forest_opt
        self.forest_ntrees = forest_ntrees
        self.forest_stepsize = forest_stepsize
        self.discretize_dense_max_buckets = discretize_dense_max_buckets
        self.discretize_dense_lamL2 = discretize_dense_lamL2
        self.discretize_dense_min_bucket_weights = discretize_dense_min_bucket_weights
        self.discretize_sparse_max_features = discretize_sparse_max_features
        self.discretize_sparse_max_buckets = discretize_sparse_max_buckets
        self.discretize_sparse_lamL2 = discretize_sparse_lamL2
        self.discretize_sparse_min_bucket_weights = discretize_sparse_min_bucket_weights
        self.discretize_sparse_min_occurences = discretize_sparse_min_occurences

        self.n_iter = n_iter
        self.calc_prob = calc_prob
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._estimators = None
        self._classes = None
        self._classes_map = {}
        self._n_classes = None
        self._n_features = None
        self._fitted = None

    def _validate_params(self, params):
        _validate_fast_rgf_params(**params)

    def _set_params_with_dependencies(self):
        if self.n_iter is None:
            if self.dtree_loss == "LS":
                self._n_iter = 10
            else:
                self._n_iter = 5
        else:
            self._n_iter = self.n_iter

        if self.n_jobs == -1:
            self._n_jobs = 0
        elif self.n_jobs < 0:
            self._n_jobs = cpu_count() + self.n_jobs + 1
        else:
            self._n_jobs = self.n_jobs

    def _get_params(self):
        return dict(dtree_max_level=self.dtree_max_level,
                    dtree_max_nodes=self.dtree_max_nodes,
                    dtree_new_tree_gain_ratio=self.dtree_new_tree_gain_ratio,
                    dtree_min_sample=self.dtree_min_sample,
                    dtree_loss=self.dtree_loss,
                    dtree_lamL1=self.dtree_lamL1,
                    dtree_lamL2=self.dtree_lamL2,
                    forest_opt=self.forest_opt,
                    forest_ntrees=self.forest_ntrees,
                    forest_stepsize=self.forest_stepsize,
                    discretize_dense_max_buckets=self.discretize_dense_max_buckets,
                    discretize_dense_lamL2=self.discretize_dense_lamL2,
                    discretize_dense_min_bucket_weights=self.discretize_dense_min_bucket_weights,
                    discretize_sparse_max_features=self.discretize_sparse_max_features,
                    discretize_sparse_max_buckets=self.discretize_sparse_max_buckets,
                    discretize_sparse_lamL2=self.discretize_sparse_lamL2,
                    discretize_sparse_min_bucket_weights=self.discretize_sparse_min_bucket_weights,
                    discretize_sparse_min_occurences=self.discretize_sparse_min_occurences,
                    n_iter=self._n_iter,
                    nthreads=self._n_jobs,
                    verbose=self.verbose)

    def _fit_binary_task(self, X, y, sample_weight, params):
        self._estimators[0] = FastRGFBinaryClassifier(**params).fit(X, y, sample_weight)

    def _fit_multiclass_task(self, X, y, sample_weight, params):
            for i, cls_num in enumerate(self._classes):
                self._classes_map[i] = cls_num
                self._estimators[i] = FastRGFBinaryClassifier(**params).fit(X,
                                                                            (y == cls_num).astype(int),
                                                                            sample_weight)


class FastRGFBinaryClassifier(utils.RGFBinaryClassifierBase):
    def save_sparse_X(self, path, X):
        utils.sparse_savetxt(path, X, including_header=False)

    def get_train_command(self):
        params = []
        params.append("forest.ntrees=%s" % self.forest_ntrees)
        params.append("forest.stepsize=%s" % self.forest_stepsize)
        params.append("forest.opt=%s" % self.forest_opt)
        params.append("discretize.dense.max_buckets=%s" % self.discretize_dense_max_buckets)
        params.append("discretize.dense.lamL2=%s" % self.discretize_dense_lamL2)
        params.append("discretize.dense.min_bucket_weights=%s" % self.discretize_dense_min_bucket_weights)
        params.append("discretize.sparse.max_features=%s" % self.discretize_sparse_max_features)
        params.append("discretize.sparse.max_buckets=%s" % self.discretize_sparse_max_buckets)
        params.append("discretize.sparse.lamL2=%s" % self.discretize_sparse_lamL2)
        params.append("discretize.sparse.min_bucket_weights=%s" % self.discretize_sparse_min_bucket_weights)
        params.append("discretize.sparse.min_occrrences=%s" % self.discretize_sparse_min_occurences)
        params.append("dtree.max_level=%s" % self.dtree_max_level)
        params.append("dtree.max_nodes=%s" % self.dtree_max_nodes)
        params.append("dtree.new_tree_gain_ratio=%s" % self.dtree_new_tree_gain_ratio)
        params.append("dtree.min_sample=%s" % self.dtree_min_sample)
        params.append("dtree.loss=%s" % self.dtree_loss)
        params.append("dtree.lamL1=%s" % self.dtree_lamL1)
        params.append("dtree.lamL2=%s" % self.dtree_lamL2)
        params.append("trn.x-file=%s" % self.train_x_loc)
        params.append("trn.y-file=%s" % self.train_y_loc)
        params.append("trn.w-file=%s" % self.train_weight_loc)
        if self.is_sparse_train_X:
            params.append("trn.x-file_format=x.sparse")
        params.append("trn.target=BINARY")
        params.append("set.nthreads=%s" % self.nthreads)
        params.append("set.verbose=%s" % self.verbose)
        params.append("model.save=%s" % self.model_file_loc)

        cmd = [utils.get_fastrgf_path() + "/forest_train"]
        cmd.extend(params)

        return cmd

    def find_model_file(self):
        if not os.path.isfile(self.model_file_loc):
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(utils.get_temp_path()))
        self.model_file = self.model_file_loc

    def get_test_command(self):
        params = []
        params.append("model.load=%s" % self.model_file)
        params.append("tst.x-file=%s" % self.test_x_loc)
        if self.is_sparse_test_X:
            params.append("tst.x-file_format=x.sparse")
        params.append("tst.target=BINARY")
        params.append("tst.output-prediction=%s" % self.pred_loc)
        params.append("set.nthreads=%s" % self.nthreads)
        params.append("set.verbose=%s" % self.verbose)

        cmd = [utils.get_fastrgf_path() + "/forest_predict"]
        cmd.extend(params)

        return cmd
