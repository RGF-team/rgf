[![Build Status](https://travis-ci.org/fukatani/rgf_python.svg?branch=master)](https://travis-ci.org/fukatani/rgf_python)

# rgf_python
Machine learning ***Regularized Greedy Forest*** wrapper for Python.

## Feature

##### Scikit learn like interface and Multi classification problem is OK.

Original rgf implementation is only available for regression and binary classification, but rgf_python is **also available for Multi classification** by "One-or-Rest" method.

ex.
```python
from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.cross_validation import StratifiedKFold
from rgf.rgf import RGFClassifier

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

rgf = RGFClassifier(max_leaf=400,
                    algorithm="RGF_Sib",
                    test_interval=100,)

# cross validation
rgf_score = 0
n_folds = 3

for train_idx, test_idx in StratifiedKFold(iris.target, n_folds):
    xs_train = iris.data[train_idx]
    y_train = iris.target[train_idx]
    xs_test = iris.data[test_idx]
    y_test = iris.target[test_idx]

    rgf.fit(xs_train, y_train)
    rgf_score += rgf.score(xs_test, y_test)

rgf_score /= n_folds
print('score: {0}'.format(rgf_score))
```

## Software Requirement

* Python (2.7 or 3.4)
* scikit-learn (0.18 or later)
* RGF(http://stat.rutgers.edu/home/tzhang/software/rgf/)

If you can't access the avove URL, alternatively, you can get RGF by downloading https://github.com/fukatani/rgf_python/releases/download/0.1.0/rgf1.2.zip.
Please see README in the zip file.


## Installation

```
git clone https://github.com/fukatani/rgf_python.git
python setup.py install
```

**And you need to edit rgf/rgf.py**

```python
## Edit this ##################################################

#Location of the RGF executable
loc_exec = 'C:\\Users\\rf\\Documents\\python\\rgf1.2\\bin\\rgf.exe'
loc_temp = 'temp/'

## End Edit ##################################################
```

You need to direct actual location of rgf execution file to 'loc_exec'.
'loc_temp' is directory for placing temp file.

##Tuning hyper parameter
You can tuning parameter as follows.

	max_leaf: Appropriate values are data-dependent and in varied from 1000 to 10000.

	test_interval: For efficiency, it must be either multiple or divisor of 100 (default of the optimization interval).

	algorithm: You can select "RGF", "RGF Opt" or "RGF Sib"

	loss: "LS", "Log" or "Expo".

	reg_depth: Must be no smaller than 1. Meant for being used with algorithm = "RGF Opt" or "RGF Sib".

	l2: Either 1, 0.1, or 0.01 often produces good results though with exponential loss (loss=Expo) and logistic loss (loss=Log) some data requires smaller values such as 1e-10 or 1e-20 Either 1, 0.1, or 0.01 often produces good results though with exponential loss (loss=Expo) and logistic loss (loss=Log) some data requires smaller values such as 1e-10 or 1e-20

	sl2: Default is equal to ls. On some data, λ/100 works well.

Detail of tuning parameter is here.
http://stat.rutgers.edu/home/tzhang/software/rgf/rgf1.2-guide.pdf

## Using at Kaggle Kernel
Now, Kaggle Kernel supports rgh_python.
Please see https://www.kaggle.com/fukatani/d/uciml/iris/classification-by-regularized-greedy-forest .

## Other

Shamelessly, many part of the implementation is based on the following. Thanks!
https://github.com/MLWave/RGF-sklearn

