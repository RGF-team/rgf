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

from rgf.sklearn import RGFClassifier, RGFRegressor
from rgf.sklearn import FastRGFClassifier, FastRGFRegressor
from rgf.utils import cleanup, get_temp_path


class RGFClassfierBaseTest(object):
    def setUp(self):
        iris = datasets.load_iris()
        rng = check_random_state(0)
        perm = rng.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(iris.data, iris.target,
                             test_size=0.2, random_state=42)

        y_str_train = np.array(self.y_train, dtype=str)
        y_str_train[y_str_train == '0'] = 'Zero'
        y_str_train[y_str_train == '1'] = 'One'
        y_str_train[y_str_train == '2'] = 'Two'
        self.y_str_train = y_str_train
        y_str_test = np.array(self.y_test, dtype=str)
        y_str_test[y_str_test == '0'] = 'Zero'
        y_str_test[y_str_test == '1'] = 'One'
        y_str_test[y_str_test == '2'] = 'Two'
        self.y_str_test = y_str_test

        self.accuracy = 0.9

    def test_classifier(self):
        clf = self.classifier_class(**self.kwargs)
        clf.fit(self.X_train, self.y_train)

        proba_sum = clf.predict_proba(self.X_test).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.y_test.shape[0]))

        score = clf.score(self.X_test, self.y_test)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_softmax_classifier(self):
        clf = self.classifier_class(calc_prob='softmax', **self.kwargs)
        clf.fit(self.X_train, self.y_train)

        proba_sum = clf.predict_proba(self.X_test).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.y_test.shape[0]))

        score = clf.score(self.X_test, self.y_test)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_bin_classifier(self):
        clf = self.classifier_class(**self.kwargs)
        bin_target_train = (self.y_train == 2).astype(int)
        bin_target_test = (self.y_test == 2).astype(int)
        clf.fit(self.X_train, bin_target_train)

        proba_sum = clf.predict_proba(self.X_test).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(bin_target_test.shape[0]))

        score = clf.score(self.X_test, bin_target_test)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_string_y(self):
        clf = self.classifier_class(**self.kwargs)

        clf.fit(self.X_train, self.y_str_train)
        y_pred = clf.predict(self.X_test)
        score = accuracy_score(self.y_str_test, y_pred)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_bin_string_y(self):
        self.accuracy = 0.75

        clf = self.classifier_class(**self.kwargs)

        bin_X_train = self.X_train[self.y_train != 0]
        bin_X_test = self.X_test[self.y_test != 0]
        y_str_train = self.y_str_train[self.y_str_train != 'Zero']
        y_str_test = self.y_str_test[self.y_str_test != 'Zero']

        clf.fit(bin_X_train, y_str_train)
        y_pred = clf.predict(bin_X_test)
        score = accuracy_score(y_str_test, y_pred)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_sklearn_integration(self):
        check_estimator(self.classifier_class)

    def test_classifier_sparse_input(self):
        clf = self.classifier_class(calc_prob='softmax', **self.kwargs)
        for sparse_format in (sparse.bsr_matrix, sparse.coo_matrix, sparse.csc_matrix,
                              sparse.csr_matrix, sparse.dia_matrix, sparse.dok_matrix, sparse.lil_matrix):
            sparse_X_train = sparse_format(self.X_train)
            sparse_X_test = sparse_format(self.X_test)
            clf.fit(sparse_X_train, self.y_train)
            score = clf.score(sparse_X_test, self.y_test)
            self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_sample_weight(self):
        clf = self.classifier_class(**self.kwargs)
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

    def test_input_arrays_shape(self):
        clf = self.classifier_class(**self.kwargs)

        n_samples = self.y_train.shape[0]
        self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train[:(n_samples - 1)])
        self.assertRaises(ValueError, clf.fit, self.X_train, self.y_train, np.ones(n_samples - 1))
        self.assertRaises(ValueError,
                          clf.fit,
                          self.X_train,
                          self.y_train,
                          np.ones((n_samples, 2)))

    def test_pickle(self):
        clf1 = self.classifier_class(**self.kwargs)
        clf1.fit(self.X_train, self.y_train)
        y_pred1 = clf1.predict(self.X_test)
        s = pickle.dumps(clf1)

        # Remove model file
        cleanup()

        clf2 = pickle.loads(s)
        y_pred2 = clf2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_joblib_pickle(self):
        clf1 = self.classifier_class(**self.kwargs)
        clf1.fit(self.X_train, self.y_train)
        y_pred1 = clf1.predict(self.X_test)
        joblib.dump(clf1, 'test_clf.pkl')

        # Remove model file
        cleanup()

        clf2 = joblib.load('test_clf.pkl')
        y_pred2 = clf2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_cleanup(self):
        clf1 = self.classifier_class(**self.kwargs)
        clf1.fit(self.X_train, self.y_train)

        clf2 = self.classifier_class(**self.kwargs)
        clf2.fit(self.X_train, self.y_train)

        self.assertNotEqual(clf1.cleanup(), 0)
        self.assertEqual(clf1.cleanup(), 0)

        for est in clf1.estimators_:
            glob_file = os.path.join(get_temp_path(), est.file_prefix + "*")
            self.assertFalse(glob.glob(glob_file))

        self.assertRaises(NotFittedError, clf1.predict, self.X_test)
        clf2.predict(self.X_test)


class TestRGFClassfier(RGFClassfierBaseTest, unittest.TestCase):
    def setUp(self):
        self.classifier_class = RGFClassifier
        self.kwargs = {}

        super(TestRGFClassfier, self).setUp()

    def test_params(self):
        clf = self.classifier_class(**self.kwargs)
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
        clf = self.classifier_class(**self.kwargs)
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

    def test_parallel_gridsearch(self):
        param_grid = dict(max_leaf=[100, 300])
        grid = GridSearchCV(self.classifier_class(n_jobs=1),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))


class TestFastRGFClassfier(RGFClassfierBaseTest, unittest.TestCase):
    def setUp(self):
        self.classifier_class = FastRGFClassifier
        self.kwargs = {}

        super(TestFastRGFClassfier, self).setUp()

    def test_params(self):
        pass

    def test_attributes(self):
        pass

    def test_sample_weight(self):
        clf = self.classifier_class(**self.kwargs)
        y_pred = clf.fit(self.X_train, self.y_train).predict_proba(self.X_test)
        y_pred_weighted = clf.fit(self.X_train,
                                  self.y_train,
                                  np.ones(self.y_train.shape[0])
                                  ).predict_proba(self.X_test)
        np.testing.assert_allclose(y_pred, y_pred_weighted)
        # TODO(fukatani): FastRGF bug?
        # does not work if weight is too small
        # weights = np.ones(self.y_train.shape[0]) * 0.01
        # weights[0] = 100
        # y_pred_weighted = clf.fit(self.X_train, self.y_train, weights).predict(self.X_test)
        # np.testing.assert_equal(y_pred_weighted, np.full(self.y_test.shape[0], self.y_test[0]))

    def test_parallel_gridsearch(self):
        param_grid = dict(forest_ntrees=[100, 300])
        grid = GridSearchCV(self.classifier_class(n_jobs=1),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_sklearn_integration(self):
        # TODO(fukatani): FastRGF bug?
        # FastRGF doesn't work if the number of sample is too small.
        # check_estimator(self.classifier_class)
        pass


class RGFRegressorBaseTest(object):
    def setUp(self):
        self.X, self.y = datasets.make_friedman1(n_samples=500,
                                                 random_state=1,
                                                 noise=1.0)
        self.X_train, self.y_train = self.X[:400], self.y[:400]
        self.X_test, self.y_test = self.X[400:], self.y[400:]

    def test_regressor(self):
        reg = self.regressor_class(**self.kwargs)
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, self.mse, "Failed with MSE = {0:.5f}".format(mse))

    def test_sklearn_integration(self):
        check_estimator(self.regressor_class)

    def test_regressor_sparse_input(self):
        reg = self.regressor_class(**self.kwargs)
        for sparse_format in (sparse.bsr_matrix, sparse.coo_matrix, sparse.csc_matrix,
                              sparse.csr_matrix, sparse.dia_matrix, sparse.dok_matrix, sparse.lil_matrix):
            X_sparse_train = sparse_format(self.X_train)
            X_sparse_test = sparse_format(self.X_test)
            reg.fit(X_sparse_train, self.y_train)
            y_pred = reg.predict(X_sparse_test)
            mse = mean_squared_error(self.y_test, y_pred)
            self.assertLess(mse, self.mse, "Failed with MSE = {0:.5f}".format(mse))

    def test_sample_weight(self):
        reg = self.regressor_class(**self.kwargs)

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

    def test_input_arrays_shape(self):
        reg = self.regressor_class(**self.kwargs)

        n_samples = self.y_train.shape[0]
        self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train[:(n_samples - 1)])
        self.assertRaises(ValueError, reg.fit, self.X_train, self.y_train, np.ones(n_samples - 1))
        self.assertRaises(ValueError,
                          reg.fit,
                          self.X_train,
                          self.y_train,
                          np.ones((n_samples, 2)))

    def test_pickle(self):
        reg1 = self.regressor_class(**self.kwargs)
        reg1.fit(self.X_train, self.y_train)
        y_pred1 = reg1.predict(self.X_test)
        s = pickle.dumps(reg1)

        # Remove model file
        cleanup()

        reg2 = pickle.loads(s)
        y_pred2 = reg2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_joblib_pickle(self):
        reg1 = self.regressor_class(**self.kwargs)
        reg1.fit(self.X_train, self.y_train)
        y_pred1 = reg1.predict(self.X_test)
        joblib.dump(reg1, 'test_reg.pkl')

        # Remove model file
        cleanup()

        reg2 = joblib.load('test_reg.pkl')
        y_pred2 = reg2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_cleanup(self):
        reg1 = self.regressor_class(**self.kwargs)
        reg1.fit(self.X_train, self.y_train)

        reg2 = self.regressor_class(**self.kwargs)
        reg2.fit(self.X_train, self.y_train)

        self.assertNotEqual(reg1.cleanup(), 0)
        self.assertEqual(reg1.cleanup(), 0)

        glob_file = os.path.join(get_temp_path(), reg1._file_prefix + "*")
        self.assertFalse(glob.glob(glob_file))

        self.assertRaises(NotFittedError, reg1.predict, self.X_test)
        reg2.predict(self.X_test)


class TestRGFRegressor(RGFRegressorBaseTest, unittest.TestCase):
    def setUp(self):
        self.regressor_class = RGFRegressor
        self.kwargs = {}

        self.mse = 2.0353275768

        super(TestRGFRegressor, self).setUp()

    def test_params(self):
        reg = self.regressor_class(**self.kwargs)

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
        reg = self.regressor_class(**self.kwargs)
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

    def test_parallel_gridsearch(self):
        param_grid = dict(max_leaf=[100, 300])
        grid = GridSearchCV(self.regressor_class(),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, self.mse, "Failed with MSE = {0:.5f}".format(mse))


class TestFastRGFRegressor(RGFRegressorBaseTest, unittest.TestCase):
    def setUp(self):
        self.regressor_class = FastRGFRegressor
        self.kwargs = {}

        self.mse = 2.5522511545

        super(TestFastRGFRegressor, self).setUp()

    def test_params(self):
        pass

    def test_attributes(self):
        pass

    def test_sample_weight(self):
        reg = self.regressor_class(**self.kwargs)

        y_pred = reg.fit(self.X_train, self.y_train).predict(self.X_test)
        y_pred_weighted = reg.fit(self.X_train,
                                  self.y_train,
                                  np.ones(self.y_train.shape[0])
                                  ).predict(self.X_test)
        np.testing.assert_allclose(y_pred, y_pred_weighted)
        # TODO(fukatani): FastRGF bug?
        # does not work if weight is too small
        # np.random.seed(42)
        # idx = np.random.choice(400, 80, replace=False)
        # self.X_train[idx] = -99999  # Add some outliers
        # y_pred_corrupt = reg.fit(self.X_train, self.y_train).predict(self.X_test)
        # mse_corrupt = mean_squared_error(self.y_test, y_pred_corrupt)
        # weights = np.ones(self.y_train.shape[0]) * 100
        # weights[idx] = 1  # Eliminate outliers
        # y_pred_weighted = reg.fit(self.X_train, self.y_train, weights).predict(self.X_test)
        # mse_fixed = mean_squared_error(self.y_test, y_pred_weighted)
        # self.assertLess(mse_fixed, mse_corrupt)

    def test_sklearn_integration(self):
        # TODO(fukatani): FastRGF bug?
        # FastRGF discretization doesn't work if the number of sample is too
        # small.
        # check_estimator(self.regressor_class)
        pass

    def test_parallel_gridsearch(self):
        param_grid = dict(forest_ntrees=[100, 500])
        grid = GridSearchCV(self.regressor_class(n_jobs=1),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, self.mse, "Failed with MSE = {0:.5f}".format(mse))

    # TODO FastRGF bug?
    # MSE with sparse input is higher than with dense one
    def test_regressor_sparse_input(self):
        self.mse = 2.5522511545
        super(TestFastRGFRegressor, self).test_regressor_sparse_input()
