from sklearn.datasets import load_boston
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from rgf.sklearn import RGFRegressor

boston = load_boston()
rng = check_random_state(42)
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

rgf = RGFRegressor(max_leaf=300,
                   algorithm="RGF_Sib",
                   test_interval=100,
                   loss="LS",
                   verbose=False)
rf = RandomForestRegressor(n_estimators=600,
                           min_samples_leaf=3,
                           max_depth=10,
                           random_state=42)

n_folds = 3

rgf_scores = cross_val_score(rgf,
                             boston.data,
                             boston.target,
                             scoring=make_scorer(mean_squared_error),
                             cv=n_folds)
rf_scores = cross_val_score(rf,
                            boston.data,
                            boston.target,
                            scoring=make_scorer(mean_squared_error),
                            cv=n_folds)

rgf_score = sum(rgf_scores)/n_folds
print('RGF Regressor MSE: {0:.5f}'.format(rgf_score))
# >>>RGF Regressor MSE: 11.79409

rf_score = sum(rf_scores)/n_folds
print('Random Forest Regressor MSE: {0:.5f}'.format(rf_score))
# >>>Random Forest Regressor MSE: 13.58614
