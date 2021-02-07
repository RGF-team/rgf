import glob
import os
import pickle
import sys
import unittest

import joblib
import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_random_state

from rgf.sklearn import RGFClassifier, RGFRegressor
from rgf.sklearn import FastRGFClassifier, FastRGFRegressor
from rgf.utils import cleanup, Config


class EstimatorBaseTest(object):
    def test_sklearn_integration(self):
        for estimator, check in check_estimator(self.estimator_class(), generate_only=True):
            check(estimator)

    def test_input_arrays_shape(self):
        est = self.estimator_class(**self.kwargs)
        n_samples = self.y_train.shape[0]
        self.assertRaises(ValueError, est.fit, self.X_train, self.y_train[:(n_samples - 1)])
        self.assertRaises(ValueError, est.fit, self.X_train, self.y_train, np.ones(n_samples - 1))
        self.assertRaises(ValueError,
                          est.fit,
                          self.X_train,
                          self.y_train,
                          np.ones((n_samples, 2)))

    def test_pickle(self):
        est1 = self.estimator_class(**self.kwargs)
        est1.fit(self.X_train, self.y_train)
        y_pred1 = est1.predict(self.X_test)
        s = pickle.dumps(est1)

        # Remove model file
        cleanup()

        est2 = pickle.loads(s)
        y_pred2 = est2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_joblib_pickle(self):
        est1 = self.estimator_class(**self.kwargs)
        est1.fit(self.X_train, self.y_train)
        y_pred1 = est1.predict(self.X_test)
        joblib.dump(est1, 'test_est.pkl')

        # Remove model file
        cleanup()

        est2 = joblib.load('test_est.pkl')
        y_pred2 = est2.predict(self.X_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_cleanup(self):
        est1 = self.estimator_class(**self.kwargs)
        est1.fit(self.X_train, self.y_train)

        est2 = self.estimator_class(**self.kwargs)
        est2.fit(self.X_train, self.y_train)

        self.assertNotEqual(est1.cleanup(), 0)
        self.assertEqual(est1.cleanup(), 0)

        for base_est in est1.estimators_:
            glob_file = os.path.join(Config().TEMP_PATH, base_est._file_prefix + "*")
            self.assertFalse(glob.glob(glob_file))

        self.assertRaises(NotFittedError, est1.predict, self.X_test)
        est2.predict(self.X_test)


class ClassifierBaseTest(EstimatorBaseTest):
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
        clf = self.estimator_class(**self.kwargs)
        clf.fit(self.X_train, self.y_train)

        proba_sum = clf.predict_proba(self.X_test).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.y_test.shape[0]))

        score = clf.score(self.X_test, self.y_test)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_softmax_classifier(self):
        clf = self.estimator_class(calc_prob='softmax', **self.kwargs)
        clf.fit(self.X_train, self.y_train)

        proba_sum = clf.predict_proba(self.X_test).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(self.y_test.shape[0]))

        score = clf.score(self.X_test, self.y_test)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_bin_classifier(self):
        clf = self.estimator_class(**self.kwargs)
        bin_target_train = (self.y_train == 2).astype(int)
        bin_target_test = (self.y_test == 2).astype(int)
        clf.fit(self.X_train, bin_target_train)

        proba_sum = clf.predict_proba(self.X_test).sum(axis=1)
        np.testing.assert_almost_equal(proba_sum, np.ones(bin_target_test.shape[0]))

        score = clf.score(self.X_test, bin_target_test)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_string_y(self):
        clf = self.estimator_class(**self.kwargs)

        clf.fit(self.X_train, self.y_str_train)
        y_pred = clf.predict(self.X_test)
        score = accuracy_score(self.y_str_test, y_pred)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_bin_string_y(self):
        self.accuracy = 0.75

        clf = self.estimator_class(**self.kwargs)

        bin_X_train = self.X_train[self.y_train != 0]
        bin_X_test = self.X_test[self.y_test != 0]
        y_str_train = self.y_str_train[self.y_str_train != 'Zero']
        y_str_test = self.y_str_test[self.y_str_test != 'Zero']

        clf.fit(bin_X_train, y_str_train)
        y_pred = clf.predict(bin_X_test)
        score = accuracy_score(y_str_test, y_pred)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_classifier_sparse_input(self):
        clf = self.estimator_class(calc_prob='softmax', **self.kwargs)
        for sparse_format in (sparse.bsr_matrix, sparse.coo_matrix, sparse.csc_matrix,
                              sparse.csr_matrix, sparse.dia_matrix, sparse.dok_matrix, sparse.lil_matrix):
            sparse_X_train = sparse_format(self.X_train)
            sparse_X_test = sparse_format(self.X_test)
            clf.fit(sparse_X_train, self.y_train)
            score = clf.score(sparse_X_test, self.y_test)
            self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))

    def test_sample_weight(self):
        clf = self.estimator_class(**self.kwargs)
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

    def test_parallel_gridsearch(self):
        self.kwargs['n_jobs'] = 1
        param_grid = dict(min_samples_leaf=[5, 10])
        grid = GridSearchCV(self.estimator_class(**self.kwargs),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(score, self.accuracy, "Failed with score = {0:.5f}".format(score))


class RegressorBaseTest(EstimatorBaseTest):
    def setUp(self):
        self.X, self.y = datasets.make_friedman1(n_samples=500,
                                                 random_state=1,
                                                 noise=1.0)
        self.X_train, self.y_train = self.X[:400], self.y[:400]
        self.X_test, self.y_test = self.X[400:], self.y[400:]

    def test_regressor(self):
        reg = self.estimator_class(**self.kwargs)
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, self.mse, "Failed with MSE = {0:.5f}".format(mse))

    def test_regressor_sparse_input(self):
        reg = self.estimator_class(**self.kwargs)
        for sparse_format in (sparse.bsr_matrix, sparse.coo_matrix, sparse.csc_matrix,
                              sparse.csr_matrix, sparse.dia_matrix, sparse.dok_matrix, sparse.lil_matrix):
            X_sparse_train = sparse_format(self.X_train)
            X_sparse_test = sparse_format(self.X_test)
            reg.fit(X_sparse_train, self.y_train)
            y_pred = reg.predict(X_sparse_test)
            mse = mean_squared_error(self.y_test, y_pred)
            self.assertLess(mse, self.mse, "Failed with MSE = {0:.5f}".format(mse))

    def test_sample_weight(self):
        reg = self.estimator_class(**self.kwargs)

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

    def test_parallel_gridsearch(self):
        param_grid = dict(min_samples_leaf=[5, 10])
        grid = GridSearchCV(self.estimator_class(**self.kwargs),
                            param_grid=param_grid, refit=True, cv=2, verbose=0, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        y_pred = grid.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, self.mse, "Failed with MSE = {0:.5f}".format(mse))


class RGFBaseTest(object):
    def test_params(self, add_valid_params=None, add_non_valid_params=None):
        add_valid_params = {} if add_valid_params is None else add_valid_params
        add_non_valid_params = {} if add_non_valid_params is None else add_non_valid_params
        est = self.estimator_class(**self.kwargs)
        valid_params = dict(max_leaf=300,
                            test_interval=100,
                            algorithm='RGF_Sib',
                            loss='Log',
                            reg_depth=1.1,
                            l2=0.1,
                            sl2=None,
                            normalize=False,
                            min_samples_leaf=0.4,
                            n_iter=None,
                            n_tree_search=2,
                            opt_interval=100,
                            learning_rate=0.4,
                            memory_policy='conservative',
                            verbose=True)
        valid_params.update(add_valid_params)
        est.set_params(**valid_params)
        est.fit(self.X_train, self.y_train)

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
        non_valid_params.update(add_non_valid_params)
        for key in non_valid_params:
            est.set_params(**valid_params)  # Reset to valid params
            est.set_params(**{key: non_valid_params[key]})  # Pick and set one non-valid parameter
            self.assertRaises(ValueError, est.fit, self.X_train, self.y_train)

    def test_attributes(self, add_attrs=None):
        self.est = self.estimator_class(**self.kwargs)
        add_attrs = [] if add_attrs is None else add_attrs
        attributes = ['n_features_', 'fitted_', 'sl2_',
                      'min_samples_leaf_', 'n_iter_', 'estimators_']
        attributes.extend(add_attrs)

        for attr in attributes:
            self.assertRaises(NotFittedError, getattr, self.est, attr)
        self.est.fit(self.X_train, self.y_train)

        self.assertEqual(self.est.n_features_, self.X_train.shape[-1])
        self.assertTrue(self.est.fitted_)
        if self.est.sl2 is None:
            self.assertEqual(self.est.sl2_, self.est.l2)
        else:
            self.assertEqual(self.est.sl2_, self.est.sl2)
        if self.est.min_samples_leaf < 1:
            self.assertLessEqual(self.est.min_samples_leaf_, 0.5 * self.X_train.shape[0])
        else:
            self.assertEqual(self.est.min_samples_leaf_, self.est.min_samples_leaf)
        if self.est.n_iter is None:
            if self.est.loss == "LS":
                self.assertEqual(self.est.n_iter_, 10)
            else:
                self.assertEqual(self.est.n_iter_, 5)
        else:
            self.assertEqual(self.est.n_iter_, self.est.n_iter)

    def test_feature_impotances(self):
        est = self.estimator_class(**self.kwargs)
        with self.assertRaises(NotFittedError):
            est.feature_importances_
        est.fit(self.X_train, self.y_train)

        fi = est.feature_importances_
        self.assertEqual(fi.shape[0], self.X_train.shape[1])
        self.assertAlmostEqual(fi.sum(), 1)

    def assertRaisesWithRegex(self, *args):
        if sys.version_info[0] == 3:
            self.assertRaisesRegex(*args)
        else:
            self.assertRaisesRegexp(*args)

    def test_warm_start(self):
        est = self.estimator_class(**self.kwargs)
        self.assertRaises(NotFittedError, est.save_model, 'model')
        est.fit(self.X_train, self.y_train)
        est.save_model('model')

        self.kwargs['init_model'] = 'no_such_file'
        new_est = self.estimator_class(**self.kwargs)
        self.assertRaisesWithRegex(Exception, "!File I/O error!",
                                   new_est.fit, self.X_train, self.y_train)

        self.kwargs['init_model'] = 'model'
        new_est = self.estimator_class(**self.kwargs)
        self.assertRaisesWithRegex(Exception, "The model given for warm-start is already "
                                              "over the requested maximum size of the models",
                                   new_est.fit, self.X_train, self.y_train)

        self.kwargs['max_leaf'] = self.new_max_leaf
        new_est = self.estimator_class(**self.kwargs)
        new_est.fit(self.X_train, self.y_train)
        self.y_pred = new_est.predict(self.X_test)


class FastRGFBaseTest(object):
    def test_params(self, add_valid_params=None, add_non_valid_params=None):
        add_valid_params = {} if add_valid_params is None else add_valid_params
        add_non_valid_params = {} if add_non_valid_params is None else add_non_valid_params
        est = self.estimator_class(**self.kwargs)
        valid_params = dict(n_estimators=50,
                            max_depth=3,
                            max_leaf=20,
                            tree_gain_ratio=0.3,
                            min_samples_leaf=0.5,
                            l1=0.6,
                            l2=100.0,
                            opt_algorithm='rgf',
                            learning_rate=0.05,
                            max_bin=150,
                            min_child_weight=9.0,
                            data_l2=9.0,
                            sparse_max_features=1000,
                            sparse_min_occurences=2,
                            n_jobs=-1,
                            verbose=True)
        valid_params.update(add_valid_params)
        est.set_params(**valid_params)
        est.fit(self.X_train, self.y_train)

        non_valid_params = dict(n_estimators=0,
                                max_depth=-3.0,
                                max_leaf=0,
                                tree_gain_ratio=1.3,
                                min_samples_leaf=0.55,
                                l1=6,
                                l2=-10.0,
                                opt_algorithm='RGF',
                                learning_rate=0.0,
                                max_bin=0.5,
                                min_child_weight='auto',
                                data_l2=None,
                                sparse_max_features=0,
                                sparse_min_occurences=-2.0,
                                n_jobs=None,
                                verbose=-3)
        for key in non_valid_params:
            est.set_params(**valid_params)  # Reset to valid params
            est.set_params(**{key: non_valid_params[key]})  # Pick and set one non-valid parameter
            self.assertRaises(ValueError, est.fit, self.X_train, self.y_train)

    def test_attributes(self, add_attrs=None):
        add_attrs = [] if add_attrs is None else add_attrs
        self.est = self.estimator_class(**self.kwargs)
        attributes = ['n_features_', 'fitted_', 'max_bin_',
                      'min_samples_leaf_', 'estimators_']
        attributes.extend(add_attrs)

        for attr in attributes:
            self.assertRaises(NotFittedError, getattr, self.est, attr)
        self.est.fit(self.X_train, self.y_train)
        self.assertEqual(self.est.n_features_, self.X_train.shape[-1])
        self.assertTrue(self.est.fitted_)
        if self.est.max_bin is None:
            if sparse.isspmatrix(self.X_train):
                self.assertEqual(self.est.max_bin_, 200)
            else:
                self.assertEqual(self.est.max_bin_, 65000)
        else:
            self.assertEqual(self.est.max_bin_, self.est.max_bin)
        if self.est.min_samples_leaf < 1:
            self.assertLessEqual(self.est.min_samples_leaf_, 0.5 * self.X_train.shape[0])
        else:
            self.assertEqual(self.est.min_samples_leaf_, self.est.min_samples_leaf)


class TestRGFClassifier(ClassifierBaseTest, RGFBaseTest, unittest.TestCase):
    def setUp(self):
        self.estimator_class = RGFClassifier
        self.kwargs = {}

        super(TestRGFClassifier, self).setUp()

    def test_params(self):
        super(TestRGFClassifier, self).test_params(add_valid_params={'calc_prob': 'sigmoid',
                                                                    'n_jobs': -1},
                                                  add_non_valid_params={'calc_prob': True,
                                                                        'n_jobs': '-1'})

    def test_attributes(self):
        super(TestRGFClassifier, self).test_attributes(add_attrs=['classes_',
                                                                 'n_classes_'])
        self.assertEqual(len(self.est.estimators_), len(np.unique(self.y_train)))
        np.testing.assert_array_equal(self.est.classes_, sorted(np.unique(self.y_train)))
        self.assertEqual(self.est.n_classes_, len(self.est.estimators_))

    def test_warm_start(self):
        self.new_max_leaf = 1050  # +50 to default value
        super(TestRGFClassifier, self).test_warm_start()
        warm_start_score = accuracy_score(self.y_test, self.y_pred)
        self.assertGreaterEqual(warm_start_score, self.accuracy,
                                "Failed with score = {0:.5f}".format(warm_start_score))


class TestFastRGFClassifier(ClassifierBaseTest, FastRGFBaseTest, unittest.TestCase):
    def setUp(self):
        self.estimator_class = FastRGFClassifier
        self.kwargs = {}

        super(TestFastRGFClassifier, self).setUp()

    def test_params(self):
        super(TestFastRGFClassifier, self).test_params(add_valid_params={'loss': 'LOGISTIC',
                                                                        'calc_prob': 'sigmoid'},
                                                      add_non_valid_params={'loss': 'LOG',
                                                                            'calc_prob': None})

    def test_attributes(self):
        super(TestFastRGFClassifier, self).test_attributes(add_attrs=['classes_',
                                                                     'n_classes_'])
        self.assertEqual(len(self.est.estimators_), len(np.unique(self.y_train)))
        np.testing.assert_array_equal(self.est.classes_, sorted(np.unique(self.y_train)))
        self.assertEqual(self.est.n_classes_, len(self.est.estimators_))

    def test_sklearn_integration(self):
        # TODO(fukatani): FastRGF bug?
        # FastRGF doesn't work if the number of sample is too small.
        # check_estimator(self.estimator_class())
        pass


class TestRGFRegressor(RegressorBaseTest, RGFBaseTest, unittest.TestCase):
    def setUp(self):
        self.estimator_class = RGFRegressor
        self.kwargs = {}

        self.mse = 2.0353275768

        super(TestRGFRegressor, self).setUp()

    def test_attributes(self):
        super(TestRGFRegressor, self).test_attributes()
        self.assertEqual(len(self.est.estimators_), 1)

    def test_abs_regressor(self):
        reg = self.estimator_class(loss="Abs")
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        self.assertLess(mae, 1.9916427774, "Failed with MAE = {0:.5f}".format(mae))

    def test_warm_start(self):
        self.new_max_leaf = 560  # +60 to default value
        super(TestRGFRegressor, self).test_warm_start()
        warm_start_score = mean_squared_error(self.y_test, self.y_pred)
        self.assertLess(warm_start_score, self.mse,
                        "Failed with MSE = {0:.5f}".format(warm_start_score))


class TestFastRGFRegressor(RegressorBaseTest, FastRGFBaseTest, unittest.TestCase):
    def setUp(self):
        self.estimator_class = FastRGFRegressor
        self.kwargs = {}

        self.mse = 2.5522511545

        super(TestFastRGFRegressor, self).setUp()

    def test_attributes(self):
        super(TestFastRGFRegressor, self).test_attributes()
        self.assertEqual(len(self.est.estimators_), 1)

    def test_parallel_gridsearch(self):
        self.kwargs['n_jobs'] = 1
        super(TestFastRGFRegressor, self).test_parallel_gridsearch()

    def test_sklearn_integration(self):
        # TODO(fukatani): FastRGF bug?
        # FastRGF doesn't work if the number of sample is too small.
        # check_estimator(self.estimator_class())
        pass
