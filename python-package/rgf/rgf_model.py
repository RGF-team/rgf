from glob import glob
from math import ceil
from shutil import copyfile

import numpy as np
from joblib import Parallel, delayed, cpu_count
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.exceptions import NotFittedError

from rgf import utils


ALGORITHMS = ("RGF", "RGF_Opt", "RGF_Sib")
LOSSES = ("LS", "Expo", "Log", "Abs")

rgf_estimator_docstring_template = \
"""
A Regularized Greedy Forest [1] {%estimator_type%}.

Tuning parameters detailed instruction:
    https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.rst#432-parameters-to-control-training

Parameters
----------
max_leaf : int, optional (default={%max_leaf_default_value%})
    Training will be terminated when the number of
    leaf nodes in the forest reaches this value.
    (Original name: max_leaf_forest.)

test_interval : int, optional (default=100)
    Test interval in terms of the number of leaf nodes.

algorithm : string ("RGF" or "RGF_Opt" or "RGF_Sib"), optional (default="RGF")
    Regularization algorithm.
    RGF: RGF with L2 regularization on leaf-only models.
    RGF Opt: RGF with min-penalty regularization.
    RGF Sib: RGF with min-penalty regularization with the sum-to-zero sibling constraints.

loss : string ("LS" or "Expo" or "Log" or "Abs"), optional (default="{%loss_default_value%}")
    Loss function.
    LS: Square loss.
    Expo: Exponential loss.
    Log: Logistic loss.
    Abs: Absolute error loss.

reg_depth : float, optional (default=1.0)
    Must be no smaller than 1.0.
    Meant for being used with algorithm="RGF Opt"|"RGF Sib".
    A larger value penalizes deeper nodes more severely.

l2 : float, optional (default=0.1)
    Used to control the degree of L2 regularization.
    (Original name: reg_L2.)

sl2 : float or None, optional (default=None)
    Override L2 regularization parameter l2
    for the process of growing the forest.
    That is, if specified, the weight correction process uses l2
    and the forest growing process uses sl2.
    If None, no override takes place and
    l2 is used throughout training.
    (Original name: reg_sL2.)

normalize : boolean, optional (default={%normalize_default_value%})
    If True, training targets are normalized
    so that the average becomes zero.
    (Original name: NormalizeTarget.)

min_samples_leaf : int or float, optional (default=10)
    Minimum number of training data points in each leaf node.
    If int, then consider min_samples_leaf as the minimum number.
    If float, then min_samples_leaf is a percentage and
    ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
    (Original name: min_pop.)

n_iter : int or None, optional (default=None)
    Number of iterations of coordinate descent to optimize weights.
    If None, 10 is used for loss="LS" and 5 for loss="Expo"|"Log".
    (Original name: num_iteration_opt.)

n_tree_search : int, optional (default=1)
    Number of trees to be searched for the nodes to split.
    The most recently grown trees are searched first.
    (Original name: num_tree_search.)

opt_interval : int, optional (default=100)
    Weight optimization interval in terms of the number of leaf nodes.
    For example, by default, weight optimization is performed
    every time approximately 100 leaf nodes are newly added to the forest.

learning_rate : float, optional (default=0.5)
    Step size of Newton updates used in coordinate descent to optimize weights.
    (Original name: opt_stepsize.)
{%calc_prob_parameter%}{%n_jobs_parameter%}
memory_policy : string ("conservative" or "generous"), optional (default="generous")
    Memory using policy.
    Generous: it runs faster using more memory by keeping the sorted orders
    of the features on memory for reuse.
    Conservative: it uses less memory at the expense of longer runtime. Try only when
    with default value it uses too much memory.

verbose : int, optional (default=0)
    Controls the verbosity of the tree building process.

init_model : None or string, optional (default=None)
    Filename of a previously saved model from which training should do warm-start.
    If model has been saved into multiple files,
    do not include numerical suffixes in the filename.

    Note
    ----
    Make sure you haven't forgotten to increase the value of the max_leaf parameter
    regarding to the specified warm-start model
    because warm-start model trees are counted in the overall number of trees.

Attributes:
-----------
estimators_ : {%estimators_property_type_desc%}
    The collection of fitted sub-estimators when `fit` is performed.
{%classes_property%}{%n_classes_property%}
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
    Learning Nonlinear Functions Using Regularized Greedy Forest
    (https://arxiv.org/abs/1109.0887).
"""


class RGFEstimatorBase(utils.CommonRGFEstimatorBase):
    def _validate_params(self,
                         max_leaf,
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
                         init_model,
                         calc_prob="sigmoid",
                         n_jobs=-1):
        if not isinstance(max_leaf, utils.INTS):
            raise ValueError(
                "max_leaf must be an integer, got {0}.".format(type(max_leaf)))
        elif max_leaf <= 0:
            raise ValueError(
                "max_leaf must be greater than 0 but was %r." % max_leaf)

        if not isinstance(test_interval, utils.INTS):
            raise ValueError(
                "test_interval must be an integer, got {0}.".format(
                    type(test_interval)))
        elif test_interval <= 0:
            raise ValueError(
                "test_interval must be greater than 0 but was %r." % test_interval)

        if not isinstance(algorithm, str):
            raise ValueError(
                "algorithm must be a string, got {0}.".format(type(algorithm)))
        elif algorithm not in ALGORITHMS:
            raise ValueError(
                "algorithm must be 'RGF' or 'RGF_Opt' or 'RGF_Sib' but was %r." % algorithm)

        if not isinstance(loss, str):
            raise ValueError(
                "loss must be a string, got {0}.".format(type(loss)))
        elif loss not in LOSSES:
            raise ValueError(
                "loss must be 'LS' or 'Expo' or 'Log' but was %r." % loss)

        if not isinstance(reg_depth, (utils.INTS, utils.FLOATS)):
            raise ValueError(
                "reg_depth must be an integer or float, got {0}.".format(
                    type(reg_depth)))
        elif reg_depth < 1:
            raise ValueError(
                "reg_depth must be no smaller than 1.0 but was %r." % reg_depth)

        if not isinstance(l2, utils.FLOATS):
            raise ValueError("l2 must be a float, got {0}.".format(type(l2)))
        elif l2 < 0:
            raise ValueError("l2 must be no smaller than 0.0 but was %r." % l2)

        if sl2 is not None and not isinstance(sl2, utils.FLOATS):
            raise ValueError(
                "sl2 must be a float or None, got {0}.".format(type(sl2)))
        elif sl2 is not None and sl2 < 0:
            raise ValueError(
                "sl2 must be no smaller than 0.0 but was %r." % sl2)

        if not isinstance(normalize, bool):
            raise ValueError(
                "normalize must be a boolean, got {0}.".format(type(normalize)))

        err_desc = "min_samples_leaf must be at least 1 or in (0, 0.5], got %r." % min_samples_leaf
        if isinstance(min_samples_leaf, utils.INTS):
            if min_samples_leaf < 1:
                raise ValueError(err_desc)
        elif isinstance(min_samples_leaf, utils.FLOATS):
            if not 0.0 < min_samples_leaf <= 0.5:
                raise ValueError(err_desc)
        else:
            raise ValueError(
                "min_samples_leaf must be an integer or float, got {0}.".format(
                    type(min_samples_leaf)))

        if n_iter is not None and not isinstance(n_iter, utils.INTS):
            raise ValueError(
                "n_iter must be an integer or None, got {0}.".format(
                    type(n_iter)))
        elif n_iter is not None and n_iter < 1:
            raise ValueError(
                "n_iter must be no smaller than 1 but was %r." % n_iter)

        if not isinstance(n_tree_search, utils.INTS):
            raise ValueError(
                "n_tree_search must be an integer, got {0}.".format(
                    type(n_tree_search)))
        elif n_tree_search < 1:
            raise ValueError(
                "n_tree_search must be no smaller than 1 but was %r." % n_tree_search)

        if not isinstance(opt_interval, utils.INTS):
            raise ValueError("opt_interval must be an integer, got {0}.".format(
                type(opt_interval)))
        elif opt_interval < 1:
            raise ValueError(
                "opt_interval must be no smaller than 1 but was %r." % opt_interval)

        if not isinstance(learning_rate, utils.FLOATS):
            raise ValueError("learning_rate must be a float, got {0}.".format(
                type(learning_rate)))
        elif learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0 but was %r." % learning_rate)

        if not isinstance(verbose, utils.INTS):
            raise ValueError(
                "verbose must be an integer, got {0}.".format(type(verbose)))
        elif verbose < 0:
            raise ValueError(
                "verbose must be no smaller than 0 but was %r." % verbose)

        if not isinstance(memory_policy, str):
            raise ValueError("memory_policy must be a string, got {0}.".format(
                type(memory_policy)))
        elif memory_policy not in ("conservative", "generous"):
            raise ValueError(
                "memory_policy must be 'conservative' or 'generous' but was %r." % memory_policy)

        if init_model is not None and not isinstance(init_model, str):
            raise ValueError(
                "init_model must be a string or None, got {0}.".format(
                    type(init_model)))

        if not isinstance(calc_prob, str):
            raise ValueError(
                "calc_prob must be a string, got {0}.".format(type(calc_prob)))
        elif calc_prob not in ("sigmoid", "softmax"):
            raise ValueError(
                "calc_prob must be 'sigmoid' or 'softmax' but was %r." % calc_prob)

        if not isinstance(n_jobs, utils.INTS):
            raise ValueError(
                "n_jobs must be an integer, got {0}.".format(type(n_jobs)))

    @property
    def sl2_(self):
        """
        The concrete regularization value for the process of growing the forest
        used in model building process.
        """
        if not hasattr(self, '_sl2'):
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._sl2

    @property
    def min_samples_leaf_(self):
        """
        Minimum number of training data points in each leaf node
        used in model building process.
        """
        if not hasattr(self, '_min_samples_leaf'):
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._min_samples_leaf

    @property
    def n_iter_(self):
        """
        Number of iterations of coordinate descent to optimize weights
        used in model building process depending on the specified loss function.
        """
        if not hasattr(self, '_n_iter'):
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        else:
            return self._n_iter

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
        res = super()._get_params()
        res.update(dict(max_leaf=self.max_leaf,
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
                        verbose=self.verbose,
                        init_model=self.init_model,
                        is_classification=is_classifier(self)))
        return res

    def _fit_binary_task(self, X, y, sample_weight, params):
        if self.n_jobs != 1 and self.verbose:
            print('n_jobs = {}, but RGFClassifier uses one CPU because classes_ is 2'.format(self.n_jobs))

        self._estimators[0] = RGFExecuter(**params).fit(X, y, sample_weight)

    def _fit_regression_task(self, X, y, sample_weight, params):
        self._estimators[0] = RGFExecuter(**params).fit(X, y, sample_weight)

    def _fit_multiclass_task(self, X, y, sample_weight, params):
        if params['init_model'] is not None:
            max_digits = len(str(len(self._classes)))
            init_model_filenames = ['{}.{}'.format(params['init_model'],
                                                   str(i + 1).zfill(max_digits)) for i in range(self._n_classes)]
        ovr_list = [None] * self._n_classes
        for i, cls_num in enumerate(self._classes):
            if params['init_model'] is not None:
                params['init_model'] = init_model_filenames[i]
            self._classes_map[i] = cls_num
            ovr_list[i] = (y == cls_num).astype(int)
            self._estimators[i] = RGFExecuter(**params)

        n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count() + self.n_jobs + 1
        substantial_n_jobs = max(n_jobs, self.n_classes_)
        if substantial_n_jobs < n_jobs and self.verbose:
            print('n_jobs = {0}, but RGFClassifier uses {1} CPUs because '
                  'classes_ is {2}'.format(n_jobs, substantial_n_jobs,
                                           self.n_classes_))

        self._estimators = Parallel(n_jobs=self.n_jobs)(delayed(utils.fit_ovr_binary)(self._estimators[i],
                                                                                      X,
                                                                                      ovr_list[i],
                                                                                      sample_weight)
                                                        for i in range(self._n_classes))

    def dump_model(self):
        """
        Dump forest information to console.

        Examples:
        ---------
        [  0], depth=0, gain=0.599606, F11, 392.8
          [  1], depth=1, gain=0.818876, F4, 0.6275
            [  3], depth=2, gain=0.806904, F5, 7.226
            [  4], depth=2, gain=0.832003, F4, 0.686
          [  2], (-0.0146), depth=1, gain=0
        Here, [ x] is order of generated, (x) is weight for leaf nodes, last value is a border.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        for est in self._estimators:
            est.dump_model()

    def save_model(self, filename):
        """
        Save model to {%file_singular_or_plural%} from which training can do warm-start in the future.
{%note%}
        Parameters
        ----------
        filename : string
            Filename to save model.

        Returns
        -------
        self : object
            Returns self.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        if len(self._estimators) > 1:
            max_digits = len(str(len(self._estimators)))
            for idx, est in enumerate(self._estimators, 1):
                est.save_model('{}.{}'.format(filename, str(idx).zfill(max_digits)))
        else:
            self._estimators[0].save_model(filename)
        return self

    @property
    def feature_importances_(self):
        """
        The feature importances.

        The importance of a feature is computed from sum of gain of each node.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise NotFittedError(utils.NOT_FITTED_ERROR_DESC)
        return np.mean([est.feature_importances_ for est in self._estimators], axis=0)


class RGFRegressor(RegressorMixin, utils.RGFRegressorMixin, RGFEstimatorBase):
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
                 verbose=0,
                 init_model=None):
        if not utils.Config().RGF_AVAILABLE:
            raise Exception('RGF estimators are unavailable for usage.')
        super().__init__()
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
        self.init_model = init_model

    _regressor_init_specific_values = {
        '{%estimator_type%}': 'regressor',
        '{%max_leaf_default_value%}': '500',
        '{%loss_default_value%}': 'LS',
        '{%normalize_default_value%}': 'True',
        '{%calc_prob_parameter%}': '',
        '{%n_jobs_parameter%}': '',
        '{%estimators_property_type_desc%}': 'one-element list of underlying regressors',
        '{%classes_property%}': '',
        '{%n_classes_property%}': ''
    }
    __doc__ = rgf_estimator_docstring_template
    for _template, _value in _regressor_init_specific_values.items():
        __doc__ = __doc__.replace(_template, _value)

    def save_model(self, filename):
        super().save_model(filename)

    _regressor_save_model_specific_values = {
        '{%file_singular_or_plural%}': 'file',
        '{%note%}': ''
    }
    save_model.__doc__ = RGFEstimatorBase.save_model.__doc__
    for _template, _value in _regressor_save_model_specific_values.items():
        save_model.__doc__ = save_model.__doc__.replace(_template, _value)


class RGFClassifier(ClassifierMixin, utils.RGFClassifierMixin, RGFEstimatorBase):
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
                 verbose=0,
                 init_model=None):
        if not utils.Config().RGF_AVAILABLE:
            raise Exception('RGF estimators are unavailable for usage.')
        super().__init__()
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
        self.calc_prob = calc_prob
        self.n_jobs = n_jobs
        self.memory_policy = memory_policy
        self.verbose = verbose
        self.init_model = init_model

    _classifier_init_specific_values = {
        '{%estimator_type%}': 'classifier',
        '{%max_leaf_default_value%}': '1000',
        '{%loss_default_value%}': 'Log',
        '{%normalize_default_value%}': 'False',
        '{%calc_prob_parameter%}': """
calc_prob : string ("sigmoid" or "softmax"), optional (default="sigmoid")
    Method of probability calculation.
""",
        '{%n_jobs_parameter%}': """
n_jobs : integer, optional (default=-1)
    The number of jobs to use for the computation.
    The substantial number of the jobs dependents on classes_.
    If classes_ = 2, the substantial max number of the jobs is one.
    If classes_ > 2, the substantial max number of the jobs is the same as classes_.
    If n_jobs = 1, no parallel computing code is used at all regardless of classes_.
    If n_jobs = -1 and classes_ >= number of CPU, all CPUs are used.
    For n_jobs = -2, all CPUs but one are used.
    For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
""",
        '{%estimators_property_type_desc%}': 'list of binary classifiers',
        '{%classes_property%}': """
classes_ : array of shape = [n_classes]
    The classes labels when `fit` is performed.
""",
        '{%n_classes_property%}': """
n_classes_ : int
    The number of classes when `fit` is performed.
"""
    }
    __doc__ = rgf_estimator_docstring_template
    for _template, _value in _classifier_init_specific_values.items():
        __doc__ = __doc__.replace(_template, _value)

    def save_model(self, filename):
        super().save_model(filename)

    _classifier_save_model_specific_values = {
        '{%file_singular_or_plural%}': 'file(s)',
        '{%note%}': """
        Note
        ----
        Due to the fact that multiclass classification problems are handled by the OvR method,
        such models are saved into multiple files with numerical suffixes,
        like filename.1, filename.2, ..., filename.n_classes.
"""
    }
    save_model.__doc__ = RGFEstimatorBase.save_model.__doc__
    for _template, _value in _classifier_save_model_specific_values.items():
        save_model.__doc__ = save_model.__doc__.replace(_template, _value)


class RGFExecuter(utils.CommonRGFExecuterBase):
    def _save_sparse_X(self, path, X):
        utils.sparse_savetxt(path, X, including_header=True)

    def _save_dense_files(self, X, y, sample_weight):
        np.savetxt(self._train_x_loc, X, delimiter=' ', fmt="%s")
        np.savetxt(self._train_y_loc, y, delimiter=' ', fmt="%s")
        if self._use_sample_weight:
            np.savetxt(self._train_weight_loc, sample_weight, delimiter=' ', fmt="%s")

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
        params.append("reg_sL2=%s" % self.sl2)
        params.append("reg_depth=%s" % self.reg_depth)
        params.append("min_pop=%s" % self.min_samples_leaf)
        params.append("num_iteration_opt=%s" % self.n_iter)
        params.append("num_tree_search=%s" % self.n_tree_search)
        params.append("opt_interval=%s" % self.opt_interval)
        params.append("opt_stepsize=%s" % self.learning_rate)
        params.append("memory_policy=%s" % self.memory_policy.title())
        params.append("model_fn_prefix=%s" % self._model_file_loc)
        if self._use_sample_weight:
            params.append("train_w_fn=%s" % self._train_weight_loc)
        if self.init_model is not None:
            params.append("model_fn_for_warmstart=%s" % self.init_model)

        cmd = (self.config.RGF_PATH, "train", ",".join(params))

        return cmd

    def _find_model_file(self):
        model_files = glob(self._model_file_loc + "*")
        if not model_files:
            raise Exception('Model learning result is not found in {0}. '
                            'Training is abnormally finished.'.format(self.config.TEMP_PATH))
        self._model_file = sorted(model_files, reverse=True)[0]

    def _get_test_command(self, is_sparse_test_X):
        params = []
        params.append("test_x_fn=%s" % self._test_x_loc)
        params.append("prediction_fn=%s" % self._pred_loc)
        params.append("model_fn=%s" % self._model_file)

        cmd = (self.config.RGF_PATH, "predict", ",".join(params))

        return cmd

    def dump_model(self):
        self._check_fitted()
        cmd = (self.config.RGF_PATH, "dump_model", "model_fn=%s" % self._model_file)
        self._execute_command(cmd, force_verbose=True)

    def save_model(self, filename):
        self._check_fitted()
        copyfile(self._model_file, filename)

    @property
    def feature_importances_(self):
        params = []
        params.append("train_x_fn=%s" % self._train_x_loc)
        params.append("feature_importances_fn=%s" % self._feature_importances_loc)
        params.append("model_fn=%s" % self._model_file)
        cmd = (self.config.RGF_PATH, "feature_importances", ",".join(params))
        self._execute_command(cmd)
        return np.loadtxt(self._feature_importances_loc)
