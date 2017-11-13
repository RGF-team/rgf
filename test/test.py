import glob
import os
import pickle
import unittest

import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_random_state

from rgf.sklearn import RGFClassifier, RGFRegressor, _cleanup, _get_temp_path


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
        clf = RGFClassifier()
        clf.fit(self.iris.data, self.iris.target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.iris.target.shape[0]))

        score = clf.score(self.iris.data, self.iris.target)
        print('Score: {0:.5f}'.format(score))
        self.assertGreater(score, 0.8, "Failed with score = {0:.5f}".format(score))

    def test_softmax_classifier(self):
        clf = RGFClassifier(calc_prob='softmax')
        clf.fit(self.iris.data, self.iris.target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.iris.target.shape[0]))

        score = clf.score(self.iris.data, self.iris.target)
        print('Score: {0:.5f}'.format(score))
        self.assertGreater(score, 0.8, "Failed with score = {0:.5f}".format(score))

    def test_bin_classifier(self):
        clf = RGFClassifier()
        bin_target = (self.iris.target == 2).astype(int)
        clf.fit(self.iris.data, bin_target)

        proba_sum = clf.predict_proba(self.iris.data).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(bin_target.shape[0]))

        score = clf.score(self.iris.data, bin_target)
        print('Score: {0:.5f}'.format(score))
        self.assertGreater(score, 0.8, "Failed with score = {0:.5f}".format(score))

    def test_string_y(self):
        clf = RGFClassifier()

        y_str = np.array(self.iris.target, dtype=str)
        y_str[y_str == '0'] = 'Zero'
        y_str[y_str == '1'] = 'One'
        y_str[y_str == '2'] = 'Two'

        clf.fit(self.iris.data, y_str)
        y_pred = clf.predict(self.iris.data)
        score = accuracy_score(y_str, y_pred)
        self.assertGreater(score, 0.95, "Failed with score = {0:.5f}".format(score))

    def test_bin_string_y(self):
        clf = RGFClassifier()

        y_str = np.array(self.iris.target, dtype=str)
        y_str[y_str == '0'] = 'Zero'
        y_str[y_str == '1'] = 'One'
        y_str[y_str == '2'] = 'Two'

        bin_X = self.iris.data[self.iris.target != 0]
        y_str = y_str[y_str != 'Zero']

        clf.fit(bin_X, y_str)
        y_pred = clf.predict(bin_X)
        score = accuracy_score(y_str, y_pred)
        self.assertGreater(score, 0.95, "Failed with score = {0:.5f}".format(score))

    def test_sklearn_integration(self):
        check_estimator(RGFClassifier)

    def test_classifier_sparse_input(self):
        clf = RGFClassifier(calc_prob='softmax')
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
                            calc_prob='sigmoid',
                            n_jobs=-1,
                            memory_policy='conservative',
                            verbose=True)
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
                                calc_prob=True,
                                n_jobs='-1',
                                memory_policy='Generos',
                                verbose=-1)
        for key in non_valid_params:
            clf.set_params(**valid_params)  # Reset to valid params
            clf.set_params(**{key: non_valid_params[key]})  # Pick and set one non-valid parametr
            self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train)

    def test_attributes(self):
        clf = RGFClassifier()
        attributes = ('estimators_', 'classes_', 'n_classes_', 'n_features_', 'fitted_',
                      'sl2_', 'min_samples_leaf_', 'n_iter_')

        for attr in attributes:
            self.assertRaises(NotFittedError, getattr, clf, attr)
        clf.fit(self.X_train, self.y_train)
        self.assertEqual(len(clf.estimators_), len(np.unique(self.y_train)))
        np.testing.assert_array_equal(clf.classes_, sorted(np.unique(self.y_train)))
        self.assertEqual(clf.n_classes_, len(clf.estimators_))
        self.assertEqual(clf.n_features_, self.X_train.shape[-1])
        self.assertTrue(clf.fitted_)
        if clf.sl2 is None:
            self.assertEqual(clf.sl2_, clf.l2)
        else:
            self.assertEqual(clf.sl2_, clf.sl2)
        if clf.min_samples_leaf < 1:
            self.assertLessEqual(clf.min_samples_leaf_, 0.5 * self.X_train.shape[0])
        else:
            self.assertEqual(clf.min_samples_leaf_, clf.min_samples_leaf)
        if clf.n_iter is None:
            if clf.loss == "LS":
                self.assertEqual(clf.n_iter_, 10)
            else:
                self.assertEqual(clf.n_iter_, 5)
        else:
            self.assertEqual(clf.n_iter_, clf.n_iter)

    def test_input_arrays_shape(self):
        clf = RGFClassifier()

        n_samples = self.y_train.shape[0]
        self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train[:(n_samples - 1)])
        self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train, np.ones(n_samples - 1))
        self.assertRaises(ValueError,
                          clf.fit,
                          self.X_train,
                          self.y_train,
                          np.ones((n_samples, 2)))

    def test_parallel_gridsearch(self):
        param_grid = dict(max_leaf=[100, 300])
        grid = GridSearchCV(RGFClassifier(n_jobs=1),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_train)
        score = accuracy_score(self.y_train, y_pred)
        self.assertGreater(score, 0.95, "Failed with score = {0:.5f}".format(score))

    def test_pickle(self):
        clf = RGFClassifier()
        clf.fit(self.X_train, self.y_train)
        y_pred1 = clf.predict(self.X_test)
        s = pickle.dumps(clf)

        # Remove model file
        _cleanup()

        reg2 = pickle.loads(s)
        y_pred2 = reg2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_joblib_pickle(self):
        clf = RGFClassifier()
        clf.fit(self.X_train, self.y_train)
        y_pred1 = clf.predict(self.X_test)
        joblib.dump(clf, 'test_clf.pkl')

        # Remove model file
        _cleanup()

        clf2 = joblib.load('test_clf.pkl')
        y_pred2 = clf2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_cleanup(self):
        clf = RGFClassifier()
        clf.fit(self.X_train, self.y_train)
        clf.cleanup()

        for est in clf.estimators_:
            glob_file = os.path.join(_get_temp_path(), est._file_prefix + "*")
            self.assertFalse(glob.glob(glob_file))


class TestRGFRegressor(unittest.TestCase):
    def setUp(self):
        # Friedman1
        self.X, self.y = datasets.make_friedman1(n_samples=500,
                                                 random_state=1,
                                                 noise=1.0)
        self.X_train, self.y_train = self.X[:400], self.y[:400]
        self.X_test, self.y_test = self.X[400:], self.y[400:]

    def test_regressor(self):
        reg = RGFRegressor()
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print("MSE: {0:.5f}".format(mse))
        self.assertLess(mse, 6.0)

    def test_sklearn_integration(self):
        check_estimator(RGFRegressor)

    def test_regressor_sparse_input(self):
        reg = RGFRegressor()
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
                            memory_policy='conservative',
                            verbose=True)
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
                                memory_policy='Generos',
                                verbose=-1)
        for key in non_valid_params:
            reg.set_params(**valid_params)  # Reset to valid params
            reg.set_params(**{key: non_valid_params[key]})  # Pick and set one non-valid parametr
            self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train)

    def test_attributes(self):
        reg = RGFRegressor()
        attributes = ('n_features_', 'fitted_', 'sl2_', 'min_samples_leaf_', 'n_iter_')

        for attr in attributes:
            self.assertRaises(NotFittedError, getattr, reg, attr)
        reg.fit(self.X_train, self.y_train)
        self.assertEqual(reg.n_features_, self.X_train.shape[-1])
        self.assertTrue(reg.fitted_)
        if reg.sl2 is None:
            self.assertEqual(reg.sl2_, reg.l2)
        else:
            self.assertEqual(reg.sl2_, reg.sl2)
        if reg.min_samples_leaf < 1:
            self.assertLessEqual(reg.min_samples_leaf_, 0.5 * self.X_train.shape[0])
        else:
            self.assertEqual(reg.min_samples_leaf_, reg.min_samples_leaf)
        if reg.n_iter is None:
            if reg.loss == "LS":
                self.assertEqual(reg.n_iter_, 10)
            else:
                self.assertEqual(reg.n_iter_, 5)
        else:
            self.assertEqual(reg.n_iter_, reg.n_iter)

    def test_input_arrays_shape(self):
        reg = RGFRegressor()

        n_samples = self.y_train.shape[0]
        self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train[:(n_samples - 1)])
        self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train, np.ones(n_samples - 1))
        self.assertRaises(ValueError,
                          reg.fit,
                          self.X_train,
                          self.y_train,
                          np.ones((n_samples, 2)))

    def test_parallel_gridsearch(self):
        param_grid = dict(max_leaf=[100, 300])
        grid = GridSearchCV(RGFRegressor(),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 6.0)

    def test_pickle(self):
        reg = RGFRegressor()
        reg.fit(self.X_train, self.y_train)
        y_pred1 = reg.predict(self.X_test)
        s = pickle.dumps(reg)

        # Remove model file
        _cleanup()

        reg2 = pickle.loads(s)
        y_pred2 = reg2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_joblib_pickle(self):
        reg = RGFRegressor()
        reg.fit(self.X_train, self.y_train)
        y_pred1 = reg.predict(self.X_test)
        joblib.dump(reg, 'test_reg.pkl')

        # Remove model file
        _cleanup()

        reg2 = joblib.load('test_reg.pkl')
        y_pred2 = reg2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_cleanup(self):
        reg = RGFRegressor()
        reg.fit(self.X_train, self.y_train)
        reg.cleanup()

        glob_file = os.path.join(_get_temp_path(), reg._file_prefix + "*")
        self.assertFalse(glob.glob(glob_file))


if __name__ == '__main__':
    unittest.main()
