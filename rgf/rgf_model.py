from __future__ import absolute_import

from glob import glob
from math import ceil
from uuid import uuid4

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed, cpu_count

from rgf import utils


ALGORITHMS = ("RGF", "RGF_Opt", "RGF_Sib")
LOSSES = ("LS", "Expo", "Log")


def validate_rgf_params(max_leaf,
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
    if not isinstance(max_leaf, utils.INTS):
        raise ValueError("max_leaf must be an integer, got {0}.".format(type(max_leaf)))
    elif max_leaf <= 0:
        raise ValueError("max_leaf must be greater than 0 but was %r." % max_leaf)

    if not isinstance(test_interval, utils.INTS):
        raise ValueError("test_interval must be an integer, got {0}.".format(type(test_interval)))
    elif test_interval <= 0:
        raise ValueError("test_interval must be greater than 0 but was %r." % test_interval)

    if not isinstance(algorithm, six.string_types):
        raise ValueError("algorithm must be a string, got {0}.".format(type(algorithm)))
    elif algorithm not in ALGORITHMS:
        raise ValueError("algorithm must be 'RGF' or 'RGF_Opt' or 'RGF_Sib' but was %r." % algorithm)

    if not isinstance(loss, six.string_types):
        raise ValueError("loss must be a string, got {0}.".format(type(loss)))
    elif loss not in LOSSES:
        raise ValueError("loss must be 'LS' or 'Expo' or 'Log' but was %r." % loss)

    if not isinstance(reg_depth, (utils.INTS, utils.FLOATS)):
        raise ValueError("reg_depth must be an integer or float, got {0}.".format(type(reg_depth)))
    elif reg_depth < 1:
        raise ValueError("reg_depth must be no smaller than 1.0 but was %r." % reg_depth)

    if not isinstance(l2, utils.FLOATS):
        raise ValueError("l2 must be a float, got {0}.".format(type(l2)))
    elif l2 < 0:
        raise ValueError("l2 must be no smaller than 0.0 but was %r." % l2)

    if not isinstance(sl2, (type(None), utils.FLOATS)):
        raise ValueError("sl2 must be a float or None, got {0}.".format(type(sl2)))
    elif sl2 is not None and sl2 < 0:
        raise ValueError("sl2 must be no smaller than 0.0 but was %r." % sl2)

    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean, got {0}.".format(type(normalize)))

    err_desc = "min_samples_leaf must be at least 1 or in (0, 0.5], got %r." % min_samples_leaf
    if isinstance(min_samples_leaf, utils.INTS):
        if min_samples_leaf < 1:
            raise ValueError(err_desc)
    elif isinstance(min_samples_leaf, utils.FLOATS):
        if not 0.0 < min_samples_leaf <= 0.5:
            raise ValueError(err_desc)
    else:
        raise ValueError("min_samples_leaf must be an integer or float, got {0}.".format(type(min_samples_leaf)))

    if not isinstance(n_iter, (type(None), utils.INTS)):
        raise ValueError("n_iter must be an integer or None, got {0}.".format(type(n_iter)))
    elif n_iter is not None and n_iter < 1:
        raise ValueError("n_iter must be no smaller than 1 but was %r." % n_iter)

    if not isinstance(n_tree_search, utils.INTS):
        raise ValueError("n_tree_search must be an integer, got {0}.".format(type(n_tree_search)))
    elif n_tree_search < 1:
        raise ValueError("n_tree_search must be no smaller than 1 but was %r." % n_tree_search)

    if not isinstance(opt_interval, utils.INTS):
        raise ValueError("opt_interval must be an integer, got {0}.".format(type(opt_interval)))
    elif opt_interval < 1:
        raise ValueError("opt_interval must be no smaller than 1 but was %r." % opt_interval)

    if not isinstance(learning_rate, utils.FLOATS):
        raise ValueError("learning_rate must be a float, got {0}.".format(type(learning_rate)))
    elif learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0 but was %r." % learning_rate)

    if not isinstance(verbose, utils.INTS):
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

    if not isinstance(n_jobs, utils.INTS):
        raise ValueError("n_jobs must be an integer, got {0}.".format(type(n_jobs)))


class RGFRegressor(utils.RGFRegressorBase):
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
        self._file_prefix = str(uuid4()) + str(utils.COUNTER.increment())
        utils.UUIDS.append(self._file_prefix)
        self._n_features = None
        self._fitted = None

    @property
    def sl2_(self):
        """
        The concrete regularization value for the process of growing the forest
        used in model building process.
        """
        if self._sl2 is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._sl2

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

    @property
    def n_iter_(self):
        """
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.
        """
        if self._n_iter is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._n_iter

    def _validate_params(self, params):
        validate_rgf_params(**params)

    def _set_params_with_dependencies(self):
        if self.sl2 is None:
            self._sl2 = self.l2
        else:
            self._sl2 = self.sl2

        if isinstance(self.min_samples_leaf, utils.FLOATS):
            self._min_samples_leaf = ceil(self.min_samples_leaf * self._n_samples)
        else:
            self._min_samples_leaf = self.min_samples_leaf

        if self.n_iter is None:
            if self.loss == "LS":
                self._n_iter = 10
            else:
                self._n_iter = 5
        else:
            self._n_iter = self.n_iter

    def _get_train_command(self):
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        if self.verbose > 5:
            params.append("Verbose_opt")  # Add some info on weight optimization
        if self.normalize:
            params.append("NormalizeTarget")
        params.append("train_x_fn=%s" % self._train_x_loc)
        params.append("train_y_fn=%s" % self._train_y_loc)
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
        params.append("model_fn_prefix=%s" % self._model_file_loc)
        params.append("train_w_fn=%s" % self._train_weight_loc)

        cmd = (utils.EXE_PATH, "train", ",".join(params))

        return cmd

    def _get_test_command(self, is_sparse_x):
        params = []
        params.append("test_x_fn=%s" % self._test_x_loc)
        params.append("prediction_fn=%s" % self._pred_loc)
        params.append("model_fn=%s" % self._model_file)

        cmd = (utils.EXE_PATH, "predict", ",".join(params))

        return cmd

    def _save_sparse_X(self, path, X):
        utils.sparse_savetxt(path, X, including_header=True)

    def _save_dense_files(self, X, y, sample_weight):
        np.savetxt(self._train_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(self._train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(self._train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

    def _find_model_file(self):
        # Find latest model location
        model_files = glob(self._model_file_loc + "*")
        if not model_files:
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(utils.TEMP_PATH))
        self._model_file = sorted(model_files, reverse=True)[0]


class RGFClassifier(utils.RGFClassifierBase):
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
        The substantial number of the jobs dependents on classes_.
        If classes_ = 2, the substantial max number of the jobs is one.
        If classes_ > 2, the substantial max number of the jobs is the same as
        classes_.
        If n_jobs = 1, no parallel computing code is used at all regardless of
        classes_.
        If n_jobs = -1 and classes_ >= number of CPU, all CPUs are used.
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
        self._classes_map = {}
        self._n_classes = None
        self._n_features = None
        self._fitted = None

    @property
    def sl2_(self):
        """
        The concrete regularization value for the process of growing the forest
        used in model building process.
        """
        if self._sl2 is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._sl2

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

    @property
    def n_iter_(self):
        """
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.
        """
        if self._n_iter is None:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._n_iter

    def _validate_params(self, params):
        validate_rgf_params(**params)

    def _set_params_with_dependencies(self):
        if self.sl2 is None:
            self._sl2 = self.l2
        else:
            self._sl2 = self.sl2

        if isinstance(self.min_samples_leaf, utils.FLOATS):
            self._min_samples_leaf = ceil(self.min_samples_leaf * self._n_samples)
        else:
            self._min_samples_leaf = self.min_samples_leaf

        if self.n_iter is None:
            if self.loss == "LS":
                self._n_iter = 10
            else:
                self._n_iter = 5
        else:
            self._n_iter = self.n_iter

    def _get_params(self):
        return dict(max_leaf=self.max_leaf,
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

    def _fit_binary_task(self, X, y, sample_weight, params):
        if self.n_jobs != 1 and self.verbose:
            print('n_jobs = {}, but RGFClassifier uses one CPU because classes_ is 2'.format(self.n_jobs))

        self._estimators[0] = RGFBinaryClassifier(**params).fit(X, y, sample_weight)

    def _fit_multiclass_task(self, X, y, sample_weight, params):
        ovr_list = [None] * self._n_classes
        for i, cls_num in enumerate(self._classes):
            self._classes_map[i] = cls_num
            ovr_list[i] = (y == cls_num).astype(int)
            self._estimators[i] = RGFBinaryClassifier(**params)

        n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count() + self.n_jobs + 1
        substantial_njobs = max(n_jobs, self.n_classes_)
        if substantial_njobs < n_jobs and self.verbose:
            print('n_jobs = {0}, but RGFClassifier uses {1} CPUs because '
                  'classes_ is {2}'.format(n_jobs, substantial_njobs,
                                           self.n_classes_))

        self._estimators = Parallel(n_jobs=self.n_jobs)(delayed(utils.fit_ovr_binary)(self._estimators[i],
                                                                                      X,
                                                                                      ovr_list[i],
                                                                                      sample_weight)
                                                        for i in range(self._n_classes))


class RGFBinaryClassifier(utils.RGFBinaryClassifierBase):
    def save_sparse_X(self, path, X):
        utils.sparse_savetxt(path, X, including_header=True)

    def save_dense_files(self, X, y, sample_weight):
        np.savetxt(self.train_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(self.train_y_loc, y, delimiter=' ', fmt="%s")
        np.savetxt(self.train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

    def get_train_command(self):
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        if self.verbose > 5:
            params.append("Verbose_opt")  # Add some info on weight optimization
        if self.normalize:
            params.append("NormalizeTarget")
        params.append("train_x_fn=%s" % self.train_x_loc)
        params.append("train_y_fn=%s" % self.train_y_loc)
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
        params.append("model_fn_prefix=%s" % self.model_file_loc)
        params.append("train_w_fn=%s" % self.train_weight_loc)

        cmd = (utils.EXE_PATH, "train", ",".join(params))

        return cmd

    def find_model_file(self):
        # Find latest model location
        model_files = glob(self.model_file_loc + "*")
        if not model_files:
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(utils.TEMP_PATH))
        self.model_file = sorted(model_files, reverse=True)[0]

    def get_test_command(self):
        params = []
        params.append("test_x_fn=%s" % self.test_x_loc)
        params.append("prediction_fn=%s" % self.pred_loc)
        params.append("model_fn=%s" % self.model_file)

        cmd = (utils.EXE_PATH, "predict", ",".join(params))

        return cmd
