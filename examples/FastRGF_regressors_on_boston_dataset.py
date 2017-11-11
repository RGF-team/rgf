from sklearn.datasets import load_boston
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from rgf.sklearn import FastRGFRegressor

boston = load_boston()
rng = check_random_state(42)
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

train_x = boston.data[:300]
test_x = boston.data[300:]
train_y = boston.target[:300]
test_y = boston.target[300:]

rgf = FastRGFRegressor()
rgf.fit(train_x, train_y)
rgf.predict(test_x)
