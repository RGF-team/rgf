import time

from sklearn.datasets import load_boston
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import RandomForestRegressor
from rgf.sklearn import FastRGFRegressor, RGFRegressor

boston = load_boston()
rng = check_random_state(42)
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

train_x = boston.data[:300]
test_x = boston.data[300:]
train_y = boston.target[:300]
test_y = boston.target[300:]

start = time.time()
reg = RGFRegressor()
reg.fit(train_x, train_y)
score = reg.score(test_x, test_y)
end = time.time()
print("RGF: {} sec".format(end - start))
print("score: {}".format(score))

start = time.time()
reg = FastRGFRegressor()
reg.fit(train_x, train_y)
score = reg.score(test_x, test_y)
end = time.time()
print("Fast RGF: {} sec".format(end - start))
print("score: {}".format(score))

start = time.time()
reg = RandomForestRegressor()
reg.fit(train_x, train_y)
score = reg.score(test_x, test_y)
end = time.time()
print("Random Forest: {} sec".format(end - start))
print("score: {}".format(score))
