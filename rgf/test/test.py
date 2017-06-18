import unittest

import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from rgf.sklearn import RGFClassifier, RGFRegressor


class TestRGFClassfier(unittest.TestCase):
    def setUp(self):
        # Iris
        iris = datasets.load_iris()
        rng = check_random_state(0)
        perm = rng.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]
        self.iris = iris

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.iris.data, self.iris.target,
                             test_size=0.2, random_state=42)

    def test_classifier(self):
        clf = RGFClassifier(prefix='clf')
        clf.fit(self.iris.data, self.iris.target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.iris.target.shape[0]))

        score = clf.score(self.iris.data, self.iris.target)
        print('Score: {0:.5f}'.format(score))
        self.assertGreater(score, 0.8, "Failed with score = {0:.5f}".format(score))

    def test_softmax_classifier(self):
        clf = RGFClassifier(prefix='clf', calc_prob='Softmax')
        clf.fit(self.iris.data, self.iris.target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.iris.target.shape[0]))

        score = clf.score(self.iris.data, self.iris.target)
        print('Score: {0:.5f}'.format(score))
        self.assertGreater(score, 0.8, "Failed with score = {0:.5f}".format(score))

    def test_bin_classifier(self):
        clf = RGFClassifier(prefix='clf')
        bin_target = (self.iris.target == 2).astype(int)
        clf.fit(self.iris.data, bin_target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(bin_target.shape[0]))

        score = clf.score(self.iris.data, bin_target)
        print('Score: {0:.5f}'.format(score))
        self.assertGreater(score, 0.8, "Failed with score = {0:.5f}".format(score))

    def test_classifier_sparse_input(self):
        clf = RGFClassifier(prefix='clf', calc_prob='Softmax')
        for sparse_format in (sparse.bsr_matrix, sparse.coo_matrix, sparse.csc_matrix,
                              sparse.csr_matrix, sparse.dia_matrix, sparse.dok_matrix, sparse.lil_matrix):
            iris_sparse = sparse_format(self.iris.data)
            clf.fit(iris_sparse, self.iris.target)
            score = clf.score(iris_sparse, self.iris.target)
            self.assertGreater(score, 0.8, "Failed with score = {0:.5f}".format(score))

    def test_sample_weight(self):
        clf = RGFClassifier()

        y_pred = clf.fit(self.X_train, self.y_train).predict_proba(self.X_test)
        y_pred_weighted = clf.fit(self.X_train,
                                  self.y_train,
                                  np.ones(self.y_train.shape[0])
                                  ).predict_proba(self.X_test)
        np.testing.assert_allclose(y_pred, y_pred_weighted)

        weights = np.ones(self.y_train.shape[0]) * np.nextafter(np.float32(0), np.float32(1))
        weights[0] = 1
        y_pred_weighted = clf.fit(self.X_train, self.y_train, weights).predict(self.X_test)
        np.testing.assert_equal(y_pred_weighted, np.full(self.y_test.shape[0], self.y_test[0]))

    def test_params(self):
        clf = RGFClassifier()

        valid_params = dict(max_leaf=300,
                            test_interval=100,
                            algorithm='RGF_Sib',
                            loss='Log',
                            reg_depth=1.1,
                            l2=0.1,
                            sl2=None,
                            normalize=False,
                            min_samples_leaf=9,
                            n_iter=None,
                            n_tree_search=2,
                            opt_interval=100,
                            learning_rate=0.4,
                            verbose=True,
                            prefix='rgf_classifier',
                            inc_prefix=True,
                            calc_prob='Sigmoid',
                            clean=True)
        clf.set_params(**valid_params)
        clf.fit(self.X_train, self.y_train)

        non_valid_params = dict(max_leaf=0,
                                test_interval=0,
                                algorithm='RGF_Test',
                                loss=True,
                                reg_depth=0.1,
                                l2=11,
                                sl2=-1.1,
                                normalize='False',
                                min_samples_leaf=0.7,
                                n_iter=11.1,
                                n_tree_search=0,
                                opt_interval=100.1,
                                learning_rate=-0.5,
                                verbose=-1,
                                prefix='',
                                inc_prefix=1,
                                calc_prob=True,
                                clean=0)
        for key in non_valid_params:
            clf.set_params(**valid_params)  # Reset to valid params
            clf.set_params(**{key: non_valid_params[key]})  # Pick and set one non-valid parametr
            self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train)

    def test_input_arrays_shape(self):
        clf = RGFClassifier()

        n_samples = self.y_train.shape[0]
        self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train[:(n_samples-1)])
        self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train, np.ones(n_samples-1))
        self.assertRaises(ValueError,
                          clf.fit,
                          self.X_train,
                          self.y_train,
                          np.ones((n_samples, 2)))


class TestRGFRegressor(unittest.TestCase):
    def setUp(self):
        # Friedman1
        self.X, self.y = datasets.make_friedman1(n_samples=500,
                                                 random_state=1,
                                                 noise=1.0)
        self.X_train, self.y_train = self.X[:400], self.y[:400]
        self.X_test, self.y_test = self.X[400:], self.y[400:]

    def test_regressor(self):
        reg = RGFRegressor(prefix='reg')
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print("MSE: {0:.5f}".format(mse))
        self.assertLess(mse, 6.0)

    def test_regressor_sparse_input(self):
        reg = RGFRegressor(prefix='reg')
        for sparse_format in (sparse.bsr_matrix, sparse.coo_matrix, sparse.csc_matrix,
                              sparse.csr_matrix, sparse.dia_matrix, sparse.dok_matrix, sparse.lil_matrix):
            X_sparse = sparse_format(self.X)
            reg.fit(X_sparse, self.y)
            y_pred = reg.predict(X_sparse)
            mse = mean_squared_error(self.y, y_pred)
            self.assertLess(mse, 6.0)

    def test_sample_weight(self):
        reg = RGFRegressor()

        y_pred = reg.fit(self.X_train, self.y_train).predict(self.X_test)
        y_pred_weighted = reg.fit(self.X_train,
                                  self.y_train,
                                  np.ones(self.y_train.shape[0])
                                  ).predict(self.X_test)
        np.testing.assert_allclose(y_pred, y_pred_weighted)

        np.random.seed(42)
        idx = np.random.choice(400, 80, replace=False)
        self.X_train[idx] = -99999  # Add some outliers
        y_pred_corrupt = reg.fit(self.X_train, self.y_train).predict(self.X_test)
        mse_corrupt = mean_squared_error(self.y_test, y_pred_corrupt)
        weights = np.ones(self.y_train.shape[0])
        weights[idx] = np.nextafter(np.float32(0), np.float32(1))  # Eliminate outliers
        y_pred_weighted = reg.fit(self.X_train, self.y_train, weights).predict(self.X_test)
        mse_fixed = mean_squared_error(self.y_test, y_pred_weighted)
        self.assertLess(mse_fixed, mse_corrupt)

    def test_params(self):
        reg = RGFRegressor()

        valid_params = dict(max_leaf=300,
                            test_interval=100,
                            algorithm='RGF_Sib',
                            loss='Log',
                            reg_depth=1.1,
                            l2=0.1,
                            sl2=None,
                            normalize=False,
                            min_samples_leaf=9,
                            n_iter=None,
                            n_tree_search=2,
                            opt_interval=100,
                            learning_rate=0.4,
                            verbose=True,
                            prefix='rgf_regressor',
                            inc_prefix=True,
                            clean=True)
        reg.set_params(**valid_params)
        reg.fit(self.X_train, self.y_train)

        non_valid_params = dict(max_leaf=0,
                                test_interval=0,
                                algorithm='RGF_Test',
                                loss=True,
                                reg_depth=0.1,
                                l2=11,
                                sl2=-1.1,
                                normalize='False',
                                min_samples_leaf=0.7,
                                n_iter=11.1,
                                n_tree_search=0,
                                opt_interval=100.1,
                                learning_rate=-0.5,
                                verbose=-1,
                                prefix='',
                                inc_prefix=1,
                                clean=0)
        for key in non_valid_params:
            reg.set_params(**valid_params)  # Reset to valid params
            reg.set_params(**{key: non_valid_params[key]})  # Pick and set one non-valid parametr
            self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train)

    def test_input_arrays_shape(self):
        reg = RGFRegressor()

        n_samples = self.y_train.shape[0]
        self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train[:(n_samples-1)])
        self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train, np.ones(n_samples-1))
        self.assertRaises(ValueError,
                          reg.fit,
                          self.X_train,
                          self.y_train,
                          np.ones((n_samples, 2)))


if __name__ == '__main__':
    unittest.main()
