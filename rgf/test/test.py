import unittest
import os
import sys


from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.utils.testing import assert_less, assert_almost_equal
import numpy as np
from rgf.rgf import RGFClassifier, RGFRegressor


class TestRGFClassfier(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        rng = check_random_state(0)
        perm = rng.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]
        self.iris = iris

    def test_classifier(self):
        clf = RGFClassifier(prefix='clf', clean=False)
        clf.fit(self.iris.data, self.iris.target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        assert_almost_equal(proba_sum, np.ones(self.iris.target.shape[0]))

        clf.clean = True
        score = clf.score(self.iris.data, self.iris.target)
        print("score: " + str(score))
        self.assertGreater(score, 0.8, "Failed with score = {0}".format(score))

    def test_softmax_classifier(self):
        clf = RGFClassifier(prefix='clf', calc_prob='Softmax', clean=False)
        clf.fit(self.iris.data, self.iris.target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        assert_almost_equal(proba_sum, np.ones(self.iris.target.shape[0]))

        clf.clean = True
        score = clf.score(self.iris.data, self.iris.target)
        print("score: " + str(score))
        self.assertGreater(score, 0.8, "Failed with score = {0}".format(score))

    def test_bin_classifier(self):
        clf = RGFClassifier(prefix='clf', clean=False)
        bin_target = (self.iris.target == 0).astype(int)
        clf.fit(self.iris.data, self.iris.target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        assert_almost_equal(proba_sum, np.ones(self.iris.target.shape[0]))

        clf.clean = True
        score = clf.score(self.iris.data, self.iris.target)
        print("score: " + str(score))
        self.assertGreater(score, 0.8, "Failed with score = {0}".format(score))

    def test_regressor(self):
        reg = RGFRegressor(prefix='reg', verbose=1)

        # Friedman1
        X, y = datasets.make_friedman1(n_samples=1200,
                                       random_state=1,
                                       noise=1.0)
        X_train, y_train = X[:200], y[:200]
        X_test, y_test = X[200:], y[200:]

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("mse: " + str(mse))
        assert_less(mse, 6.0)


if __name__ == '__main__':
    unittest.main()
