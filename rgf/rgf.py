import os
import platform
import subprocess
from glob import glob
import numpy as np
from math import ceil

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import NotFittedError

sys_name = platform.system()
WINDOWS = 'Windows'
LINUX = 'Linux'

## Edit this ##################################################
if sys_name == WINDOWS:
    #Location of the RGF executable
    loc_exec = 'C:\\Program Files\\RGF\\bin\\rgf.exe'
    #Location for RGF temp files
    loc_temp = 'temp/'
    default_exec = 'rgf.exe'
elif sys_name == LINUX:
	#Location of the RGF executable
    loc_exec = '/opt/rgf1.2/bin/rgf'
    #Location for RGF temp files
    loc_temp = '/tmp/rgf'
    default_exec = 'rgf'
## End Edit ##################################################

def is_executable_response(path):
    try:
        subprocess.check_output([path, "train"])
        return True
    except:
        return False

# validate path
if is_executable_response(default_exec):
	loc_exec = default_exec
elif not os.path.isfile(loc_exec):
    raise Exception('{0} is not executable file. Please set '
                    'loc_exec to RGF execution file.'.format(loc_exec))
elif not os.access(loc_exec, os.X_OK):
    raise Exception('{0} cannot be accessed. Please set '
                    'loc_exec to RGF execution file.'.format(loc_exec))
elif is_executable_response(loc_exec):
	pass
else:
    raise Exception('{0} does not exist or {1} is not in the "PATH" variable.'.format(loc_exec,
    	                                                                              default_exec))


def sigmoid(x):
    """x : array-like
    output : array-like
    """
    return 1. / (1.+ np.exp(-x))

class RGFClassifier(BaseEstimator, ClassifierMixin):
    """A Regularized Greedy Forest [1] classifier.

    Tuning parameters detailed instruction:
        http://tongzhang-ml.org/software/rgf/rgf1.2-guide.pdf

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

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    prefix : string, optional (default="rgf_classifier")
        Used as a prefix for RGF output temp file.

    inc_prefix : boolean, optional (default=True)
        If True, auto increment for numbering temp file is enable.

    calc_prob : string ("Sigmoid" or "Softmax"), optional (default="Sigmoid")
        Method of probability calculation.

    clean : boolean, optional (default=True)
        If True, remove temp files before fitting.
        If False, previous leaning result will be loaded.

    Reference
    ---------
    [1] Rie Johnson and Tong Zhang.
        Learning Nonlinear Functions Using Regularized Greedy Forest
    """
    instance_count = 0
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
                 verbose=0,
                 prefix="rgf_classifier",
                 inc_prefix=True,
                 calc_prob='Sigmoid',
                 clean=True):
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
        self.verbose = verbose
        self.prefix = prefix
        self.inc_prefix = inc_prefix
        self.file_prefix = prefix
        if inc_prefix:
            self.file_prefix = prefix + str(RGFClassifier.instance_count)
            RGFClassifier.instance_count += 1
        self.calc_prob = calc_prob
        self.clean = clean        
        self.fitted = False

    def fit(self, X, y, sample_weight=None):
        """Build a RGF Classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification).

        sample_weight : array-like, shape = [n_samples] or None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = sorted(np.unique(y))
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ <= 2:
            self.estimator = RGFBinaryClassifier(max_leaf=self.max_leaf,
                                                 test_interval=self.test_interval,
                                                 algorithm=self.algorithm,
                                                 loss=self.loss,
                                                 reg_depth=self.reg_depth,
                                                 l2=self.l2,
                                                 sl2=self.sl2,
                                                 normalize=self.normalize,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 n_iter=self.n_iter,
                                                 n_tree_search=self.n_tree_search,
                                                 opt_interval=self.opt_interval,
                                                 learning_rate=self.learning_rate,
                                                 verbose=self.verbose,
                                                 prefix=self.prefix,
                                                 inc_prefix=self.inc_prefix,
                                                 clean=self.clean)
            self.estimator.fit(X, y)
        else:
            self.estimators = [None] * self.n_classes_
            for i, cls_num in enumerate(self.classes_):
                y_one_or_rest = (y == cls_num).astype(int)
                prefix = "{0}_c{1}".format(self.prefix, i)
                self.estimators[i] = RGFBinaryClassifier(max_leaf=self.max_leaf,
                                                         test_interval=self.test_interval,
                                                         algorithm=self.algorithm,
                                                         loss=self.loss,
                                                         reg_depth=self.reg_depth,
                                                         l2=self.l2,
                                                         sl2=self.sl2,
                                                         normalize=self.normalize,
                                                         min_samples_leaf=self.min_samples_leaf,
                                                         n_iter=self.n_iter,
                                                         n_tree_search=self.n_tree_search,
                                                         opt_interval=self.opt_interval,
                                                         learning_rate=self.learning_rate,
                                                         verbose=self.verbose,
                                                         prefix=prefix,
                                                         inc_prefix=True,
                                                         clean=self.clean)
                self.estimators[i].fit(X, y_one_or_rest)
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes].
            The class probabilities of the input samples.
        """

        if self.n_classes_ <= 2:
            proba = self.estimator.predict_proba(X)
            proba = sigmoid(proba)
            proba = np.c_[1-proba, proba]
        else:
            proba = np.zeros((X.shape[0], self.n_classes_))
            for i, clf in enumerate(self.estimators):
                class_proba = clf.predict_proba(X)
                proba[:, i] = class_proba

            #In honest I don't understand which is better
            #softmax or normalized sigmoid for calc probability.
            if self.calc_prob == "Sigmoid":
                proba = sigmoid(proba)
                normalizer = proba.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba /= normalizer
            else:
                proba = softmax(proba)
        return proba

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the StackedClassifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class RGFBinaryClassifier(BaseEstimator, ClassifierMixin):
    """RGF Binary Classifier.
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
                 verbose=0,
                 prefix="rgf_classifier",
                 inc_prefix=True,
                 clean=True):

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
        self.verbose = verbose
        self.prefix = prefix
        self.inc_prefix = inc_prefix
        self.file_prefix = prefix
        self.clean = clean
        self.fitted = False
        if not os.path.isdir(loc_temp):
            os.mkdir(loc_temp)
        if not os.access(loc_temp, os.W_OK):
            raise Exception('{0} is not writable directory. Please set '
                            'loc_temp to writable directory'.format(loc_temp))

    #Fitting/training the model to target variables
    def fit(self, X, y, sample_weight=None):
        #Clean temp directory
        if self.clean:
            model_glob = loc_temp + os.sep + "*"

            for fn in glob(model_glob):
                if "predictions.txt" in fn or self.prefix in fn or "train.data." in fn or "test.data." in fn:
                    os.remove(fn)

		#Store the train set into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.x"), X, delimiter=' ', fmt="%s")

        #convert 1 to 1, 0 to -1
        y = 2*y - 1
	    #Store the targets into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.y"), y, delimiter=' ', fmt="%s")

        if sample_weight is not None:
            #Store the weights into RGF format
            np.savetxt(os.path.join(loc_temp, "train.data.weight"), sample_weight, delimiter=' ', fmt="%s")

        if self.sl2 is None:
            self.sl2 = self.l2

        if type(self.min_samples_leaf) == float:
            self.min_samples_leaf = ceil(self.min_samples_leaf * np.asarray(X).shape[0])

        if self.n_iter is None:
            if self.loss == "LS":
                self.n_iter = 10
            else:
                self.n_iter = 5

	    #format train command
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        if self.normalize:
            params.append("NormalizeTarget")
        params.append("train_x_fn=%s"%os.path.join(loc_temp, "train.data.x"))
        params.append("train_y_fn=%s"%os.path.join(loc_temp, "train.data.y"))
        params.append("algorithm=%s"%self.algorithm)
        params.append("loss=%s"%self.loss)
        params.append("max_leaf_forest=%s"%self.max_leaf)
        params.append("test_interval=%s"%self.test_interval)
        params.append("reg_L2=%s"%self.l2)
        params.append("reg_sL2=%s"%self.sl2)
        params.append("reg_depth=%s"%self.reg_depth)
        params.append("min_pop=%s"%self.min_samples_leaf)
        params.append("num_iteration_opt=%s"%self.n_iter)
        params.append("num_tree_search=%s"%self.n_tree_search)
        params.append("opt_interval=%s"%self.opt_interval)
        params.append("opt_stepsize=%s"%self.learning_rate)
        params.append("model_fn_prefix=%s"%os.path.join(loc_temp, self.file_prefix))
        if sample_weight is not None:
            params.append("train_w_fn=%s"%os.path.join(loc_temp, "train.data.weight"))

        cmd = [loc_exec, "train", ",".join(params)]

        #train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)
        self.fitted = True
        return self

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        #Store the test set into RGF format
        np.savetxt(os.path.join(loc_temp, "test.data.x"), X, delimiter=' ', fmt="%s")

        #Find latest model location
        model_glob = loc_temp + os.sep + self.file_prefix + "*"
        if not glob(model_glob):
            raise Exception('Model learning result is not found in {0}. This is rgf_python error.'.format(loc_temp))
        latest_model_loc = sorted(glob(model_glob), reverse=True)[0]

        #Format test command
        params = []
        params.append("test_x_fn=%s"%os.path.join(loc_temp, "test.data.x"))
        params.append("prediction_fn=%s"%os.path.join(loc_temp, "predictions.txt"))
        params.append("model_fn=%s"%latest_model_loc)

        cmd = [loc_exec, "predict", ",".join(params)]

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        y_pred = np.loadtxt(os.path.join(loc_temp, "predictions.txt"))
        return y_pred


class RGFRegressor(BaseEstimator, RegressorMixin):
    """A Regularized Greedy Forest [1] regressor.

    Tuning parameters detailed instruction:
        http://tongzhang-ml.org/software/rgf/rgf1.2-guide.pdf

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

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    prefix : string, optional (default="rgf_regressor")
        Used as a prefix for RGF output temp file.

    inc_prefix : boolean, optional (default=True)
        If True, auto increment for numbering temp file is enable.

    clean : boolean, optional (default=True)
        If True, remove temp files before fitting.
        If False, previous leaning result will be loaded.

    Reference
    ---------
    [1] Rie Johnson and Tong Zhang.
        Learning Nonlinear Functions Using Regularized Greedy Forest
    """
    instance_count = 0
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
                 verbose=0,
                 prefix="rgf_regressor",
                 inc_prefix=True,
                 clean=True):
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
        self.verbose = verbose
        self.prefix = prefix
        self.inc_prefix = inc_prefix
        self.file_prefix = prefix
        if inc_prefix:
            self.file_prefix = prefix + str(RGFRegressor.instance_count)
            RGFRegressor.instance_count += 1
        self.clean = clean
        if not os.path.isdir(loc_temp):
            os.mkdir(loc_temp)
        if not os.access(loc_temp, os.W_OK):
            raise Exception('{0} is not writable directory. Please set loc_temp to writable directory'.format(loc_temp))
        self.fitted = False

    def fit(self, X, y, sample_weight=None):
        """Build a RGF Regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape of shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (real numbers in regression).

        sample_weight : array-like, shape = [n_samples] or None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns self.
        """
        #Clean temp directory
        if self.clean:
            model_glob = loc_temp + os.sep + "*"

            for fn in glob(model_glob):
                if "predictions.txt" in fn or self.prefix in fn or "train.data." in fn or "test.data." in fn:
                    os.remove(fn)

        #Store the train set into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.x"), X, delimiter=' ', fmt="%s")
        #Store the targets into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.y"), y, delimiter=' ', fmt="%s")
        if sample_weight is not None:
            #Store the weights into RGF format
            np.savetxt(os.path.join(loc_temp, "train.data.weight"), sample_weight, delimiter=' ', fmt="%s")

        if self.sl2 is None:
            self.sl2 = self.l2

        if type(self.min_samples_leaf) == float:
            self.min_samples_leaf = ceil(self.min_samples_leaf * np.asarray(X).shape[0])

        if self.n_iter is None:
            if self.loss == "LS":
                self.n_iter = 10
            else:
                self.n_iter = 5

		#format train command
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        if self.normalize:
            params.append("NormalizeTarget")
        params.append("train_x_fn=%s"%os.path.join(loc_temp, "train.data.x"))
        params.append("train_y_fn=%s"%os.path.join(loc_temp, "train.data.y"))
        params.append("algorithm=%s"%self.algorithm)
        params.append("loss=%s"%self.loss)
        params.append("max_leaf_forest=%s"%self.max_leaf)
        params.append("test_interval=%s"%self.test_interval)
        params.append("reg_L2=%s"%self.l2)
        params.append("reg_sL2=%s"%self.sl2)
        params.append("reg_depth=%s"%self.reg_depth)
        params.append("min_pop=%s"%self.min_samples_leaf)
        params.append("num_iteration_opt=%s"%self.n_iter)
        params.append("num_tree_search=%s"%self.n_tree_search)
        params.append("opt_interval=%s"%self.opt_interval)
        params.append("opt_stepsize=%s"%self.learning_rate)
        params.append("model_fn_prefix=%s"%os.path.join(loc_temp, self.file_prefix))
        if sample_weight is not None:
            params.append("train_w_fn=%s"%os.path.join(loc_temp, "train.data.weight"))

        cmd = [loc_exec, "train", ",".join(params)]

        #train
        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)
        self.fitted = True
        return self

    def predict(self, X):
        """The predicted value of an input sample is a vote by the RGFRegressor.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        if not self.fitted:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        #Store the test set into RGF format
        np.savetxt(os.path.join(loc_temp, "test.data.x"), X, delimiter=' ', fmt="%s")

        #Find latest model location
        model_glob = loc_temp + os.sep + self.file_prefix + "*"
        if not glob(model_glob):
            raise Exception('Model learning result is not found in {0}. This is rgf_python error.'.format(loc_temp))
        latest_model_loc = sorted(glob(model_glob), reverse=True)[0]

        #Format test command
        params = []
        params.append("test_x_fn=%s"%os.path.join(loc_temp, "test.data.x"))
        params.append("prediction_fn=%s"%os.path.join(loc_temp, "predictions.txt"))
        params.append("model_fn=%s"%latest_model_loc)

        cmd = [loc_exec, "predict", ",".join(params)]

        output = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True).communicate()

        if self.verbose:
            for k in output:
                print(k)

        y_pred = np.loadtxt(os.path.join(loc_temp, "predictions.txt"))
        return y_pred
