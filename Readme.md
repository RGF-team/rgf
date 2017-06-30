[![Build Status Travis](https://travis-ci.org/fukatani/rgf_python.svg?branch=master)](https://travis-ci.org/fukatani/rgf_python)
[![Build Status AppVeyor](https://ci.appveyor.com/api/projects/status/vpanb9hnncjr16hn/branch/master?svg=true)](https://ci.appveyor.com/project/fukatani/rgf-python)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/fukatani/rgf_python/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/rgf_python.svg)](https://pypi.python.org/pypi/rgf_python/)
[![PyPI Version](https://badge.fury.io/py/rgf_python.svg)](https://badge.fury.io/py/rgf_python)
<!-- [![PyPI Version](https://img.shields.io/pypi/v/rgf_python.svg)](https://pypi.python.org/pypi/rgf_python/) # Reserve link for PyPI in case of bugs at fury.io -->

# rgf_python
The wrapper of machine learning algorithm ***Regularized Greedy Forest (RGF)*** for Python.

## Features

##### Scikit-learn interface and possibility of usage for multi-label classification problem.

Original RGF implementation is available only for regression and binary classification, but rgf_python is also available for **multi-label classification** by "One-vs-Rest" method.

Example:
```python
from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rgf.sklearn import RGFClassifier

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

rgf = RGFClassifier(max_leaf=400,
                    algorithm="RGF_Sib",
                    test_interval=100,
                    verbose=True)

n_folds = 3

rgf_scores = cross_val_score(rgf,
                             iris.data,
                             iris.target,
                             cv=StratifiedKFold(n_folds))

rgf_score = sum(rgf_scores)/n_folds
print('RGF Classfier score: {0:.5f}'.format(rgf_score))
```
More examples could be found [here](https://github.com/fukatani/rgf_python/tree/master/examples).

## Software Requirements

* Python (2.7 or >= 3.4)
* scikit-learn (>= 0.18)
* RGF C++ ([link](http://tongzhang-ml.org/software/rgf/index.html))

If you can't access the above URL, alternatively, you can get RGF C++ by downloading it from this [page](https://github.com/fukatani/rgf_python/releases/download/0.2.0/rgf1.2.zip).
Please see README in the zip file to build RGF executional.


## Installation

From [PyPI](https://pypi.python.org/pypi/rgf_python) using `pip`:
```
pip install rgf_python
```

or from [GitHub](https://github.com/fukatani/rgf_python):
```
git clone https://github.com/fukatani/rgf_python.git
python setup.py install
```

You have to place RGF execution file into directory which is included in environmental variable **'PATH'**.
Alternatively, you may specify actual location of RGF execution file and directory for placing temp files by corresponding flags in configuration file `.rgfrc`, which you should create into your home directory. The default values are platform dependent: for Windows `exe_location=$HOME/rgf.exe`, `temp_location=$HOME/temp/rgf` and for others `exe_location=$HOME/rgf`, `temp_location=/tmp/rgf`. Here is the example of `.rgfrc`:

```
exe_location=C:/Program Files/RGF/bin/rgf.exe
temp_location=C:/Program Files/RGF/temp
```

## Tuning Hyper-parameters
You can tune hyper-parameters as follows.

* _max_leaf_: Appropriate values are data-dependent and usually varied from 1000 to 10000.

* _test_interval_: For efficiency, it must be either multiple or divisor of 100 (default value of the optimization interval).

* _algorithm_: You can select "RGF", "RGF Opt" or "RGF Sib".

* _loss_: You can select "LS", "Log" or "Expo".

* _reg_depth_: Must be no smaller than 1. Meant for being used with _algorithm_ = "RGF Opt" or "RGF Sib".

* _l2_: Either 1, 0.1, or 0.01 often produces good results though with exponential loss (_loss_ = "Expo") and logistic loss (_loss_ = "Log"), some data requires smaller values such as 1e-10 or 1e-20.

* _sl2_: Default value is equal to _l2_. On some data, _l2_/100 works well.

* _normalize_: If turned on, training targets are normalized so that the average becomes zero.

* _min_samples_leaf_: Smaller values may slow down training. Too large values may degrade model accuracy.

* _n_iter_: Number of iterations of coordinate descent to optimize weights.

* _n_tree_search_: Number of trees to be searched for the nodes to split. The most recently grown trees are searched first.

* _opt_interval_: Weight optimization interval in terms of the number of leaf nodes.

* _learning_rate_: Step size of Newton updates used in coordinate descent to optimize weights.

Detailed instruction of tuning hyper-parameters is [here](http://tongzhang-ml.org/software/rgf/rgf1.2-guide.pdf).

## Using at Kaggle Kernel
Now, Kaggle Kernel supports rgf_python. Please see [this page](https://www.kaggle.com/fukatani/d/uciml/iris/classification-by-regularized-greedy-forest).

## Other
Shamelessly, much part of the implementation is based on the following [code](https://github.com/MLWave/RGF-sklearn). Thanks!
