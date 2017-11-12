import time

from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import GradientBoostingClassifier
from rgf.sklearn import RGFClassifier, FastRGFClassifier

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

train_x = iris.data[:300]
test_x = iris.data[300:]
train_y = iris.target[:300]
test_y = iris.target[300:]

# start = time.time()
# clf = RGFClassifier()
# clf.fit(iris.data, iris.target)
# score = clf.score(iris.data, iris.target)
# end = time.time()
# print("RGF: {} sec".format(end - start))
# print("score: {}".format(score))

start = time.time()
clf = FastRGFClassifier()
clf.fit(iris.data, iris.target)
score = clf.score(iris.data, iris.target)
end = time.time()
print("Fast RGF: {} sec".format(end - start))
print("score: {}".format(score))

start = time.time()
clf = GradientBoostingClassifier()
clf.fit(iris.data, iris.target)
score = clf.score(iris.data, iris.target)
end = time.time()
print("Fast Random Forest: {} sec".format(end - start))
print("score: {}".format(score))
