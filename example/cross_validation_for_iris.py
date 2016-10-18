import os
import sys

from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from rgf.rgf import RGFClassifier

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

rgf = RGFClassifier(max_leaf=400,
                    algorithm="RGF_Sib",
                    test_interval=100,
                    verbose=True)
gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.01, subsample=0.6)

# cross validation
rgf_score = 0
gb_score = 0
n_folds = 3

for train_idx, test_idx in StratifiedKFold(n_folds).split(iris.data, iris.target):
    xs_train = iris.data[train_idx]
    y_train = iris.target[train_idx]
    xs_test = iris.data[test_idx]
    y_test = iris.target[test_idx]

    rgf.fit(xs_train, y_train)
    rgf_score += rgf.score(xs_test, y_test)
    gb.fit(xs_train, y_train)
    gb_score += gb.score(xs_test, y_test)

rgf_score /= n_folds
print('RGF Classfier score: {0}'.format(rgf_score))
gb_score /= n_folds
print('Gradient Boosting Classfier score: {0}'.format(gb_score))
