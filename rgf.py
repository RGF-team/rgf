
import os
import subprocess
from glob import glob
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

## Edit this ##################################################

#Location of the RGF executable
loc_exec='C:\\Users\\rf\\Documents\\python\\rgf1.2\\bin\\rgf.exe'
loc_temp='temp/'

def sigmoid(x) :
    return 1/(1+np.exp(-x))

class RGFClassifier(BaseEstimator, ClassifierMixin):
    instance_count = 0
    def __init__(self,
                 verbose=0,
                 max_leaf=500,
                 test_interval=100,
                 algorithm="RGF",
                 loss="Log",
                 depth=1,
                 l2=0.1,
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
        self.inc_prefix = inc_prefix
        if inc_prefix:
            self.file_prefix = prefix + str(RGFClassifier.instance_count)
            RGFClassifier.instance_count += 1
        self.depth = depth
        self.l2 = l2
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
                                                 depth=self.depth,
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
                                     depth=self.depth,
                                     l2=self.l2,
                                     prefix=prefix,
                                     inc_prefix=False,
                                     clean=self.clean)
                self.estimators[i].fit(X, y_one_or_rest)

    def predict_proba(self, X):
        if self.n_classes_ <= 2:
            self.estimator.predict_proba(X)
        else:
            proba = np.zeros((X.shape[0], self.n_classes_))
            for i, clf in enumerate(self.estimators):
                proba[:, i] = clf.predict_proba(X)[:, 1]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer
            return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class RGFBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 verbose=0,
                 max_leaf=500,
                 test_interval=100,
                 algorithm="RGF",
                 loss="Log",
                 depth=1,
                 l2=0.1,
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
        self.depth = depth
        self.l2 = l2
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
        params.append("reg_depth=%s"%self.depth)
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

        y_pred = sigmoid(np.loadtxt(os.path.join(loc_temp, "predictions.txt")))
        y_pred = np.c_[1-y_pred, y_pred]

    	#Clean temp directory
    	if self.clean:
    		model_glob = loc_temp + os.sep + "*"

    		for fn in glob(model_glob):
    			if "predictions.txt" in fn or "model-" in fn or "train.data." in fn or "test.data." in fn:
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
		return params

class RGFRegressor(BaseEstimator, RegressorMixin):
    instance_count = 0
    def __init__(self,
                 verbose=0,
                 max_leaf=500,
                 test_interval=100,
                 algorithm="RGF",
                 loss="LS",
                 l2=0.1,
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
        if inc_prefix:
            self.file_prefix = prefix + str(RGFRegressor.instance_count)
            RGFRegressor.instance_count += 1
        self.l2 = l2
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
        params.append("model_fn_prefix=%s"%os.path.join(loc_temp, self.file_prefix))

        cmd = "%s train %s 2>&1"%(loc_exec,",".join(params))

        #train
        output = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True).communicate()

        if self.verbose:
            for k in output:
                print k

        return self

    def predict(self, X):
        #Store the test set into RGF format
        np.savetxt(os.path.join(loc_temp, "test.data.x"), X, delimiter=' ', fmt="%s")

        #Find latest model location
        model_glob = loc_temp + os.sep + self.file_prefix + "*"
        latest_model_loc = sorted(glob(model_glob),reverse=True)[0]

        #Format test command
        params = []
        params.append("test_x_fn=%s"%os.path.join(loc_temp, "test.data.x"))
        params.append("prediction_fn=%s"%os.path.join(loc_temp, "predictions.txt"))
        params.append("model_fn=%s"%latest_model_loc)
        cmd = "%s predict %s"%(loc_exec,",".join(params)) # 2>&1

        output = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE,shell=True).communicate()

        if self.verbose:
            for k in output:
                print k

		y_pred = np.loadtxt(os.path.join(loc_temp, "predictions.txt"))

		#Clean temp directory
		if self.clean:
			model_glob = loc_temp + os.sep + "*"

			for fn in glob(model_glob):
				if "predictions.txt" in fn or "model-" in fn or "train.data." in fn or "test.data." in fn:
					os.remove(fn)
		return y_pred