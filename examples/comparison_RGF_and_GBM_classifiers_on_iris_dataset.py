from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from rgf.sklearn import RGFClassifier

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

rgf = RGFClassifier(max_leaf=400,
                    algorithm="RGF_Sib",
                    test_interval=100,
                    verbose=True)
gb = GradientBoostingClassifier(n_estimators=20,
                                learning_rate=0.01,
                                subsample=0.6,
                                random_state=rng)

n_folds = 3

rgf_scores = cross_val_score(rgf,
                             iris.data,
                             iris.target,
                             cv=StratifiedKFold(n_folds))

gb_scores = cross_val_score(gb,
                            iris.data,
                            iris.target,
                            cv=StratifiedKFold(n_folds))

rgf_score = sum(rgf_scores)/n_folds
print('RGF Classfier score: {0:.5f}'.format(rgf_score))
# >>>RGF Classfier score: 0.95997

gb_score = sum(gb_scores)/n_folds
print('Gradient Boosting Classfier score: {0:.5f}'.format(gb_score))
# >>>Gradient Boosting Classfier score: 0.95997
