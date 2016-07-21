import os
import subprocess
from glob import glob
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

## Edit this ##################################################

#Location of the RGF executable
loc_exec = 'C:\\Users\\rf\\Documents\\python\\rgf1.2\\bin\\rgf.exe'
loc_temp = 'temp/'

## End Edit

def sigmoid(x):
    """x : array-like
    output : array-like

    """
    return 1. / (1.+ np.exp(-x))

def softmax(x):
    """x : array-like
    output : array-like

    """
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

class RGFClassifier(BaseEstimator, ClassifierMixin):
    """A Regularized Greedy Forest[1] classifier.

    Tunig parameter Detail :
        http://stat.rutgers.edu/home/tzhang/software/rgf/rgf1.2-guide.pdf

    Parameters
    ----------

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    max_leaf : int, optional (default=1000)
        Training will be terminated when the number of
        leaf nodes in the forest reaches this value.

    test_interval : int, optional (default=100)
        Test interval in terms of the number of leaf nodes.

    algorithm : string, "RGF" or "RGF_Opt" or "RGF_Sib"
        Regularization algorithm.

    loss : "LS" or "Expo" or "Log".
        Loss function.

    reg_depth : float, (default=1)
        Meant for being used with algorithm=RGF Opt|RGF Sib.
        A larger value penalizes deeper nodes more severely.

    l2 : float, (default=0.1)
        Used to control the degree of L2 regularization.

    sl2 : float, (default=None)
        Override L2 regularization parameter l2
        for the process of growing the forest.

    prefix : string, (default="model")
        Used as a prefix for rgf output temp file.

    inc_prefix : boolean, (default=False)
        If Trur, auto increment for numbering temp file is enable.

    calc_prob : String, "Sigmoid" or "Softmax"
        Method of probability calculation.

    clean : boolean, (default=True)
        If True, remove temp files after prediction.
        If False previous leaning result will be loaded.

    Reference.
    [1] Rie Johnson and Tong Zhang. Learning nonlinear functions using regularized greedy forest

    """
    instance_count = 0
    def __init__(self,
                 verbose=0,
                 max_leaf=1000,
                 test_interval=100,
                 algorithm="RGF",
                 loss="Log",
                 reg_depth=1,
                 l2=0.1,
                 sl2=None,
                 prefix="model",
                 inc_prefix=False,
                 calc_prob='Sigmoid',
                 clean=True):
        self.verbose = verbose
        self.max_leaf = max_leaf
        self.algorithm = algorithm
        self.loss = loss
        self.test_interval = test_interval
        self.prefix = prefix
        self.file_prefix = prefix
        self.inc_prefix = inc_prefix
        if inc_prefix:
            self.file_prefix = prefix + str(RGFClassifier.instance_count)
            RGFClassifier.instance_count += 1
        self.reg_depth = reg_depth
        self.l2 = l2
        self.calc_prob = calc_prob
        if sl2 is None:
            self.sl2 = l2
        else:
            self.sl2 = sl2
        self.clean = clean

    def fit(self, X, y):
        self.classes_ = sorted(np.unique(y))
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ <= 2:
            self.estimator = RGFBinaryClassifier(verbose=self.verbose,
                                                 max_leaf=self.max_leaf,
                                                 test_interval=self.test_interval,
                                                 algorithm=self.algorithm,
                                                 loss=self.loss,
                                                 reg_depth=self.reg_depth,
                                                 l2=self.l2,
                                                 prefix=self.prefix,
                                                 inc_prefix=self.inc_prefix,
                                                 clean=self.clean)
            self.estimator.fit(X, y)
        else:
            self.estimators = [None] * self.n_classes_
            for i, cls_num in enumerate(self.classes_):
                y_one_or_rest = (y == cls_num).astype(int)
                prefix = "{0}_c{1}".format(self.prefix, i)
                self.estimators[i] = RGFBinaryClassifier(verbose=self.verbose,
                                     max_leaf=self.max_leaf,
                                     test_interval=self.test_interval,
                                     algorithm=self.algorithm,
                                     loss=self.loss,
                                     reg_depth=self.reg_depth,
                                     l2=self.l2,
                                     prefix=prefix,
                                     inc_prefix=False,
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
            proba = np.c_[1-proba, proba]
        else:
            proba = np.zeros((X.shape[0], self.n_classes_))
            for i, clf in enumerate(self.estimators):
                class_proba = clf.predict_proba(X)
                proba[:, i] = class_proba

            #In honest I don't understand which is better
            # softmax or normalized sigmoid for calc probability.
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
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.

        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_params(self, deep=False):
        params = {}
        params["verbose"] = self.verbose
        params["max_leaf"] = self.max_leaf
        params["algorithm"] = self.algorithm
        params["loss"] = self.loss
        params["test_interval"] = self.test_interval
        params["prefix"] = self.prefix
        params["l2"] = self.l2
        params["sl2"] = self.sl2
        params["reg_depth"] = self.reg_depth
        return params


class RGFBinaryClassifier(BaseEstimator, ClassifierMixin):
    """RGF Binary Classifier.
    Don't instantiate this class directly.
    RGFBinaryClassifier should be instantiated only by RGFClassifier.

    """
    def __init__(self,
                 verbose=0,
                 max_leaf=500,
                 test_interval=100,
                 algorithm="RGF",
                 loss="Log",
                 reg_depth=1,
                 l2=0.1,
                 sl2=None,
                 prefix="model",
                 inc_prefix=False,
                 clean=True):
        self.verbose = verbose
        self.max_leaf = max_leaf
        self.algorithm = algorithm
        self.loss = loss
        self.test_interval = test_interval
        self.prefix = prefix
        self.file_prefix = prefix
        self.reg_depth = reg_depth
        self.l2 = l2
        if sl2 is None:
            self.sl2 = l2
        else:
            self.sl2 = sl2
        self.clean = clean
        if not os.path.isdir(loc_temp):
            os.mkdir(loc_temp)

	#Fitting/training the model to target variables
    def fit(self, X, y):
		#Store the train set into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.x"), X, delimiter=' ', fmt="%s")

        #convert 1 to 1, 0 to -1
        y = 2*y - 1

		#Store the targets into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.y"), y, delimiter=' ', fmt="%s")

		#format train command
        params = []
        if self.verbose > 0:
            params.append("Verbose")
        params.append("train_x_fn=%s"%os.path.join(loc_temp, "train.data.x"))
        params.append("train_y_fn=%s"%os.path.join(loc_temp, "train.data.y"))
        params.append("algorithm=%s"%self.algorithm)
        params.append("loss=%s"%self.loss)
        params.append("max_leaf_forest=%s"%self.max_leaf)
        params.append("test_interval=%s"%self.test_interval)
        params.append("reg_L2=%s"%self.l2)
        params.append("reg_sL2=%s"%self.sl2)
        params.append("reg_depth=%s"%self.reg_depth)
        params.append("model_fn_prefix=%s"%os.path.join(loc_temp, self.file_prefix))

        cmd = "%s train %s 2>&1"%(loc_exec, ",".join(params))

        #train
        output = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True).communicate()

        if self.verbose:
            for k in output:
                print k
        return self

    def predict_proba(self, X):
        #Store the test set into RGF format
        np.savetxt(os.path.join(loc_temp, "test.data.x"), X, delimiter=' ', fmt="%s")

        #Find latest model location
        model_glob = loc_temp + os.sep + self.file_prefix + "*"
        latest_model_loc = sorted(glob(model_glob), reverse=True)[0]

        #Format test command
        params = []
        params.append("test_x_fn=%s"%os.path.join(loc_temp, "test.data.x"))
        params.append("prediction_fn=%s"%os.path.join(loc_temp, "predictions.txt"))
        params.append("model_fn=%s"%latest_model_loc)
        cmd = "%s predict %s 2>&1"%(loc_exec, ",".join(params))

        output = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True).communicate()

        if self.verbose:
            for k in output:
                print k

        y_pred = np.loadtxt(os.path.join(loc_temp, "predictions.txt"))

        #Clean temp directory
        if self.clean:
            model_glob = loc_temp + os.sep + "*"

            for fn in glob(model_glob):
                if "predictions.txt" in fn or self.prefix in fn or "train.data." in fn or "test.data." in fn:
                    os.remove(fn)

        return y_pred


class RGFRegressor(BaseEstimator, RegressorMixin):
    instance_count = 0
    def __init__(self,
                 verbose=0,
                 max_leaf=500,
                 test_interval=100,
                 algorithm="RGF",
                 loss="LS",
                 l2=0.1,
                 sl2=None,
                 prefix="model",
                 reg_depth=1,
                 inc_prefix=False,
                 clean=True):
        self.verbose = verbose
        self.max_leaf = max_leaf
        self.algorithm = algorithm
        self.loss = loss
        self.test_interval = test_interval
        self.prefix = prefix
        self.file_prefix = prefix
        if inc_prefix:
            self.file_prefix = prefix + str(RGFRegressor.instance_count)
            RGFRegressor.instance_count += 1
        self.reg_depth = reg_depth
        self.l2 = l2
        if sl2 is None:
            self.sl2 = l2
        else:
            self.sl2 = sl2
        self.clean = clean

    #Fitting/training the model to target variables
    def fit(self, X, y):
        #Store the train set into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.x"), X, delimiter=' ', fmt="%s")
        #Store the targets into RGF format
        np.savetxt(os.path.join(loc_temp, "train.data.y"), y, delimiter=' ', fmt="%s")

		#format train command
        params = []
        if self.verbose > 0:
            params.append("Verbose")
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
        params.append("model_fn_prefix=%s"%os.path.join(loc_temp, self.file_prefix))

        cmd = "%s train %s 2>&1"%(loc_exec, ",".join(params))

        #train
        output = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True).communicate()

        if self.verbose:
            for k in output:
                print k

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
        #Store the test set into RGF format
        np.savetxt(os.path.join(loc_temp, "test.data.x"), X, delimiter=' ', fmt="%s")

        #Find latest model location
        model_glob = loc_temp + os.sep + self.file_prefix + "*"
        latest_model_loc = sorted(glob(model_glob), reverse=True)[0]

        #Format test command
        params = []
        params.append("test_x_fn=%s"%os.path.join(loc_temp, "test.data.x"))
        params.append("prediction_fn=%s"%os.path.join(loc_temp, "predictions.txt"))
        params.append("model_fn=%s"%latest_model_loc)
        cmd = "%s predict %s"%(loc_exec, ",".join(params)) # 2>&1

        output = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True).communicate()

        if self.verbose:
            for k in output:
                print k

        y_pred = np.loadtxt(os.path.join(loc_temp, "predictions.txt"))

        #Clean temp directory
        if self.clean:
            model_glob = loc_temp + os.sep + "*"

            for fn in glob(model_glob):
                if "predictions.txt" in fn or self.prefix in fn or "train.data." in fn or "test.data." in fn:
                    os.remove(fn)
        return y_pred

    def get_params(self, deep=False):
        params = {}
        params["verbose"] = self.verbose
        params["max_leaf"] = self.max_leaf
        params["algorithm"] = self.algorithm
        params["loss"] = self.loss
        params["test_interval"] = self.test_interval
        params["prefix"] = self.prefix
        params["l2"] = self.l2
        params["sl2"] = self.sl2
        params["reg_depth"] = self.reg_depth
        return params
