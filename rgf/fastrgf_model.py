from __future__ import absolute_import

import os
from uuid import uuid4

import numpy as np
from sklearn.externals.joblib import cpu_count

from rgf import utils


def validate_fast_rgf_params(**kwargs):
    pass


class FastRGFRegressor(utils.RGFRegressorBase):
    """
    A Fast Regularized Greedy Forest regressor by Tong Zhang.
    See https://github.com/baidu/fast_rgf.

    This function is alpha version.
    The part of the function may be not tested, not documented and not
    unstabled. API can be changed in the future.

    Parameters
    ----------
    n_estimators : int, optional (default=500)
        The number of trees in the forest.
        (Original name: forest.ntrees.)

    max_depth : int, optional (default=6)
        Maximum tree depth.
        (Original name: dtree.max_level.)

    max_leaf : int, optional (default=50)
        Maximum number of leaf nodes in best-first search.
        (Original name: dtree.max_nodes.)

    tree_gain_ratio : float, optional (default=1.0)
        New tree is created when leaf-nodes gain < this value * estimated gain
        of creating new tree.
        (Original name: dtree.new_tree_gain_ratio.)

    min_samples_leaf : int or float, optional (default=5)
        Minimum number of training data points in each leaf node.
        If int, then consider min_samples_leaf as the minimum number.
        If float, then min_samples_leaf is a percentage and
        ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
        (Original name: dtree.min_sample.)

    l1 : float, optional (default=1.0)
        Used to control the degree of L1 regularization.
        (Original name: dtree.lamL1.)

    l2 : float, optional (default=1000.0)
        Used to control the degree of L2 regularization.
        (Original name: dtree.lamL2.)

    opt_algorithm : string ("rgf" or "epsilon-greedy"), optional (default="rgf")
        Optimization method for training forest.
        (Original name: forest.opt.)

    learning_rate : float, optional (default=0.001)
        Step size of epsilon-greedy boosting.
        Meant for being used with opt_algorithm="epsilon-greedy".
        (Original name: forest.stepsize.)

    max_bin : int or None, optional (default=None)
        Maximum number of discretized values (bins).
        If None, 65000 is used for dense data and 200 for sparse data.
        (Original name: discretize.(sparse/dense).max_buckets.)

    min_child_weight : float, optional (default=5.0)
        Minimum sum of data weights for each discretized value (bin).
        (Original name: discretize.(sparse/dense).min_bucket_weights.)

    data_l2 : float, optional (default=2.0)
        Used to control the degree of L2 regularization for discretization.
        (Original name: discretize.(sparse/dense).lamL2.)

    sparse_max_features : int, optional (default=80000)
        Maximum number of selected features.
        Meant for being used with sparse data.
        (Original name: discretize.sparse.max_features.)

    sparse_min_occurences : int, optional (default=5)
        Minimum number of occurrences for a feature to be selected.
        Meant for being used with sparse data.
        (Original name: discretize.sparse.min_occrrences.)

    n_jobs : integer, optional (default=-1)
        The number of jobs to run in parallel for both fit and predict.
        If -1, all CPUs are used.
        If -2, all CPUs but one are used.
        If < -1, (n_cpus + 1 + n_jobs) are used.
        (Original name: set.nthreads.)

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
        (Original name: set.verbose.)

    Attributes:
    -----------
    n_features_ : int
        The number of features when `fit` is performed.

    fitted_ : boolean
        Indicates whether `fit` is performed.

    max_bin_ : int
        The concrete maximum number of discretized values (bins)
        used in model building process for given data.

    min_samples_leaf_ : int
        Minimum number of training data points in each leaf node
        used in model building process.
    """
    # TODO(fukatani): Test
    def __init__(self,
                 n_estimators=500,
                 max_depth=6,
                 max_leaf=50,
                 tree_gain_ratio=1.0,
                 min_samples_leaf=5,
                 l1=1.0,
                 l2=1000.0,
                 opt_algorithm="rgf",
                 learning_rate=0.001,
                 max_bin=None,
                 min_child_weight=5.0,
                 data_l2=2.0,
                 sparse_max_features=80000,
                 sparse_min_occurences=5,
                 n_jobs=-1,
                 verbose=0):
        if not utils.FASTRGF_AVAILABLE:
            raise Exception('FastRGF is not installed correctly.')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf = max_leaf
        self.tree_gain_ratio = tree_gain_ratio
        self.min_samples_leaf = min_samples_leaf
        self.l1 = l1
        self.l2 = l2
        self.opt_algorithm = opt_algorithm
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self._max_bin = None
        self.min_child_weight = min_child_weight
        self.data_l2 = data_l2
        self.sparse_max_features = sparse_max_features
        self.sparse_min_occurences = sparse_min_occurences
        self.n_jobs = n_jobs
        self._n_jobs = None
        self.verbose = verbose

        self._file_prefix = str(uuid4()) + str(utils.COUNTER.increment())
        utils.UUIDS.append(self._file_prefix)
        self._n_features = None
        self._fitted = None

    @property
    def max_bin_(self):
        """
        The concrete maximum number of discretized values (bins)
        used in model building process for given data.
        """
        if self._max_bin is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._max_bin

    @property
    def min_samples_leaf_(self):
        """
        Minimum number of training data points in each leaf node
        used in model building process.
        """
        if self._min_samples_leaf is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._min_samples_leaf

    def _validate_params(self, params):
        validate_fast_rgf_params(**params)

    def _set_params_with_dependencies(self):
        if self.max_bin is None:
            if self._is_sparse_train_X:
                self._max_bin = 200
            else:
                self._max_bin = 65000
        else:
            self._max_bin = self.max_bin

        if isinstance(self.min_samples_leaf, utils.FLOATS):
            self._min_samples_leaf = ceil(self.min_samples_leaf * self._n_samples)
        else:
            self._min_samples_leaf = self.min_samples_leaf

        if self.n_jobs == -1:
            self._n_jobs = 0
        elif self.n_jobs < 0:
            self._n_jobs = cpu_count() + self.n_jobs + 1
        else:
            self._n_jobs = self.n_jobs

    def _get_train_command(self):
        params = []
        params.append("forest.ntrees=%s" % self.n_estimators)
        params.append("forest.stepsize=%s" % self.learning_rate)
        params.append("forest.opt=%s" % self.opt_algorithm)
        params.append("dtree.max_level=%s" % self.max_depth)
        params.append("dtree.max_nodes=%s" % self.max_leaf)
        params.append("dtree.new_tree_gain_ratio=%s" % self.tree_gain_ratio)
        params.append("dtree.min_sample=%s" % self._min_samples_leaf)
        params.append("dtree.lamL1=%s" % self.l1)
        params.append("dtree.lamL2=%s" % self.l2)
        if self._is_sparse_train_X:
            params.append("discretize.sparse.max_features=%s" % self.sparse_max_features)
            params.append("discretize.sparse.max_buckets=%s" % self._max_bin)
            params.append("discretize.sparse.lamL2=%s" % self.data_l2)
            params.append("discretize.sparse.min_bucket_weights=%s" % self.min_child_weight)
            params.append("discretize.sparse.min_occrrences=%s" % self.sparse_min_occurences)
            params.append("trn.x-file_format=x.sparse")
            params.append("trn.y-file=%s" % self._train_y_loc)
            params.append("trn.w-file=%s" % self._train_weight_loc)
        else:
            params.append("discretize.dense.max_buckets=%s" % self._max_bin)
            params.append("discretize.dense.lamL2=%s" % self.data_l2)
            params.append("discretize.dense.min_bucket_weights=%s" % self.min_child_weight)
            params.append("trn.x-file_format=w.y.x")
        params.append("trn.x-file=%s" % self._train_x_loc)
        params.append("trn.target=REAL")
        params.append("set.nthreads=%s" % self._n_jobs)
        params.append("set.verbose=%s" % self.verbose)
        params.append("model.save=%s" % self._model_file_loc)

        cmd = [os.path.join(utils.FASTRGF_PATH, "forest_train")]
        cmd.extend(params)

        return cmd

    def _get_test_command(self, is_sparse_x):
        params = []
        params.append("model.load=%s" % self._model_file)
        params.append("tst.x-file=%s" % self._test_x_loc)
        if is_sparse_x:
            params.append("tst.x-file_format=x.sparse")
        params.append("tst.target=REAL")
        params.append("tst.output-prediction=%s" % self._pred_loc)
        params.append("set.nthreads=%s" % self._n_jobs)
        params.append("set.verbose=%s" % self.verbose)

        cmd = [os.path.join(utils.FASTRGF_PATH, "forest_predict")]
        cmd.extend(params)

        return cmd

    def _save_sparse_X(self, path, X):
        utils.sparse_savetxt(path, X, including_header=False)

    def _save_dense_files(self, X, y, sample_weight):
        self._train_x_loc = self._train_x_loc[:-2]
        np.savetxt(self._train_x_loc, np.c_[sample_weight, y, X], delimiter=' ', fmt="%s")

    def _find_model_file(self):
        if not os.path.isfile(self._model_file_loc):
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(utils.TEMP_PATH))
        self._model_file = self._model_file_loc


class FastRGFClassifier(utils.RGFClassifierBase):
    """
    A Fast Regularized Greedy Forest classifier by Tong Zhang.
    See https://github.com/baidu/fast_rgf.

    This function is alpha version.
    The part of the function may be not tested, not documented and not
    unstabled. API can be changed in the future.

    Parameters
    ----------
    n_estimators : int, optional (default=500)
        The number of trees in the forest.
        (Original name: forest.ntrees.)

    max_depth : int, optional (default=6)
        Maximum tree depth.
        (Original name: dtree.max_level.)

    max_leaf : int, optional (default=50)
        Maximum number of leaf nodes in best-first search.
        (Original name: dtree.max_nodes.)

    tree_gain_ratio : float, optional (default=1.0)
        New tree is created when leaf-nodes gain < this value * estimated gain
        of creating new tree.
        (Original name: dtree.new_tree_gain_ratio.)

    min_samples_leaf : int or float, optional (default=5)
        Minimum number of training data points in each leaf node.
        If int, then consider min_samples_leaf as the minimum number.
        If float, then min_samples_leaf is a percentage and
        ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
        (Original name: dtree.min_sample.)

    loss : string ("LS" or "MODLS" or "LOGISTIC"), optional (default="LS")
        Loss function.
        LS: Least squares loss.
        MODLS: Modified least squares loss.
        LOGISTIC: Logistic loss.
        (Original name: dtree.loss.)

    l1 : float, optional (default=1.0)
        Used to control the degree of L1 regularization.
        (Original name: dtree.lamL1.)

    l2 : float, optional (default=1000.0)
        Used to control the degree of L2 regularization.
        (Original name: dtree.lamL2.)

    opt_algorithm : string ("rgf" or "epsilon-greedy"), optional (default="rgf")
        Optimization method for training forest.
        (Original name: forest.opt.)

    learning_rate : float, optional (default=0.001)
        Step size of epsilon-greedy boosting.
        Meant for being used with opt_algorithm="epsilon-greedy".
        (Original name: forest.stepsize.)

    max_bin : int or None, optional (default=None)
        Maximum number of discretized values (bins).
        If None, 65000 is used for dense data and 200 for sparse data.
        (Original name: discretize.(sparse/dense).max_buckets.)

    min_child_weight : float, optional (default=5.0)
        Minimum sum of data weights for each discretized value (bin).
        (Original name: discretize.(sparse/dense).min_bucket_weights.)

    data_l2 : float, optional (default=2.0)
        Used to control the degree of L2 regularization for discretization.
        (Original name: discretize.(sparse/dense).lamL2.)

    sparse_max_features : int, optional (default=80000)
        Maximum number of selected features.
        Meant for being used with sparse data.
        (Original name: discretize.sparse.max_features.)

    sparse_min_occurences : int, optional (default=5)
        Minimum number of occurrences for a feature to be selected.
        Meant for being used with sparse data.
        (Original name: discretize.sparse.min_occrrences.)

    calc_prob : string ("sigmoid" or "softmax"), optional (default="sigmoid")
        Method of probability calculation.

    n_jobs : integer, optional (default=-1)
        The number of jobs to run in parallel for both fit and predict.
        If -1, all CPUs are used.
        If -2, all CPUs but one are used.
        If < -1, (n_cpus + 1 + n_jobs) are used.
        (Original name: set.nthreads.)

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
        (Original name: set.verbose.)

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

    max_bin_ : int
        The concrete maximum number of discretized values (bins)
        used in model building process for given data.

    min_samples_leaf_ : int
        Minimum number of training data points in each leaf node
        used in model building process.
    """
    # TODO(fukatani): Test
    def __init__(self,
                 n_estimators=500,
                 max_depth=6,
                 max_leaf=50,
                 tree_gain_ratio=1.0,
                 min_samples_leaf=5,
                 loss="LS",
                 l1=1.0,
                 l2=1000.0,
                 opt_algorithm="rgf",
                 learning_rate=0.001,
                 max_bin=None,
                 min_child_weight=5.0,
                 data_l2=2.0,
                 sparse_max_features=80000,
                 sparse_min_occurences=5,
                 calc_prob="sigmoid",
                 n_jobs=-1,
                 verbose=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf = max_leaf
        self.tree_gain_ratio = tree_gain_ratio
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        self.l1 = l1
        self.l2 = l2
        self.opt_algorithm = opt_algorithm
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self._max_bin = None
        self.min_child_weight = min_child_weight
        self.data_l2 = data_l2
        self.sparse_max_features = sparse_max_features
        self.sparse_min_occurences = sparse_min_occurences

        self.calc_prob = calc_prob
        self.n_jobs = n_jobs
        self._n_jobs = None
        self.verbose = verbose

        self._estimators = None
        self._classes = None
        self._classes_map = {}
        self._n_classes = None
        self._n_features = None
        self._fitted = None

    @property
    def max_bin_(self):
        """
        The concrete maximum number of discretized values (bins)
        used in model building process for given data.
        """
        if self._max_bin is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._max_bin

    @property
    def min_samples_leaf_(self):
        """
        Minimum number of training data points in each leaf node
        used in model building process.
        """
        if self._min_samples_leaf is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._min_samples_leaf

    def _validate_params(self, params):
        validate_fast_rgf_params(**params)

    def _set_params_with_dependencies(self):
        if self.max_bin is None:
            if self._is_sparse_train_X:
                self._max_bin = 200
            else:
                self._max_bin = 65000
        else:
            self._max_bin = self.max_bin

        if isinstance(self.min_samples_leaf, utils.FLOATS):
            self._min_samples_leaf = ceil(self.min_samples_leaf * self._n_samples)
        else:
            self._min_samples_leaf = self.min_samples_leaf

        if self.n_jobs == -1:
            self._n_jobs = 0
        elif self.n_jobs < 0:
            self._n_jobs = cpu_count() + self.n_jobs + 1
        else:
            self._n_jobs = self.n_jobs

    def _get_params(self):
        return dict(max_depth=self.max_depth,
                    max_leaf=self.max_leaf,
                    tree_gain_ratio=self.tree_gain_ratio,
                    min_samples_leaf=self._min_samples_leaf,
                    loss=self.loss,
                    l1=self.l1,
                    l2=self.l2,
                    opt_algorithm=self.opt_algorithm,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_bin=self._max_bin,
                    data_l2=self.data_l2,
                    min_child_weight=self.min_child_weight,
                    sparse_max_features=self.sparse_max_features,
                    sparse_min_occurences=self.sparse_min_occurences,
                    n_jobs=self._n_jobs,
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

    def save_dense_files(self, X, y, sample_weight):
        self.train_x_loc = self.train_x_loc[:-2]
        np.savetxt(self.train_x_loc, np.c_[sample_weight, y, X], delimiter=' ', fmt="%s")

    def get_train_command(self):
        params = []
        params.append("forest.ntrees=%s" % self.n_estimators)
        params.append("forest.stepsize=%s" % self.learning_rate)
        params.append("forest.opt=%s" % self.opt_algorithm)
        params.append("dtree.max_level=%s" % self.max_depth)
        params.append("dtree.max_nodes=%s" % self.max_leaf)
        params.append("dtree.new_tree_gain_ratio=%s" % self.tree_gain_ratio)
        params.append("dtree.min_sample=%s" % self.min_samples_leaf)
        params.append("dtree.loss=%s" % self.loss)
        params.append("dtree.lamL1=%s" % self.l1)
        params.append("dtree.lamL2=%s" % self.l2)
        if self.is_sparse_train_X:
            params.append("discretize.sparse.max_features=%s" % self.sparse_max_features)
            params.append("discretize.sparse.max_buckets=%s" % self.max_bin)
            params.append("discretize.sparse.lamL2=%s" % self.data_l2)
            params.append("discretize.sparse.min_bucket_weights=%s" % self.min_child_weight)
            params.append("discretize.sparse.min_occrrences=%s" % self.sparse_min_occurences)
            params.append("trn.x-file_format=x.sparse")
            params.append("trn.y-file=%s" % self.train_y_loc)
            params.append("trn.w-file=%s" % self.train_weight_loc)
        else:
            params.append("discretize.dense.max_buckets=%s" % self.max_bin)
            params.append("discretize.dense.lamL2=%s" % self.data_l2)
            params.append("discretize.dense.min_bucket_weights=%s" % self.min_child_weight)
            params.append("trn.x-file_format=w.y.x")
        params.append("trn.x-file=%s" % self.train_x_loc)
        params.append("trn.target=BINARY")
        params.append("set.nthreads=%s" % self.n_jobs)
        params.append("set.verbose=%s" % self.verbose)
        params.append("model.save=%s" % self.model_file_loc)

        cmd = [os.path.join(utils.FASTRGF_PATH, "forest_train")]
        cmd.extend(params)

        return cmd

    def find_model_file(self):
        if not os.path.isfile(self.model_file_loc):
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(utils.TEMP_PATH))
        self.model_file = self.model_file_loc

    def get_test_command(self):
        params = []
        params.append("model.load=%s" % self.model_file)
        params.append("tst.x-file=%s" % self.test_x_loc)
        if self.is_sparse_test_X:
            params.append("tst.x-file_format=x.sparse")
        params.append("tst.target=BINARY")
        params.append("tst.output-prediction=%s" % self.pred_loc)
        params.append("set.nthreads=%s" % self.n_jobs)
        params.append("set.verbose=%s" % self.verbose)

        cmd = [os.path.join(utils.FASTRGF_PATH, "forest_predict")]
        cmd.extend(params)

        return cmd
