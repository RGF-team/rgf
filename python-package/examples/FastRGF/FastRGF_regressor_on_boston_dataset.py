import time

from sklearn.datasets import load_diabetes
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import RandomForestRegressor
from rgf.sklearn import FastRGFRegressor, RGFRegressor

diabetes = load_diabetes()
rng = check_random_state(42)
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

train_x = diabetes.data[:300]
test_x = diabetes.data[300:]
train_y = diabetes.target[:300]
test_y = diabetes.target[300:]

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
print("FastRGF: {} sec".format(end - start))
print("score: {}".format(score))

start = time.time()
reg = RandomForestRegressor(n_estimators=100)
reg.fit(train_x, train_y)
score = reg.score(test_x, test_y)
end = time.time()
print("Random Forest: {} sec".format(end - start))
print("score: {}".format(score))
