import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from PreTrainingChain.PreTrainingChain import ChainClassfier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier, XGBRegressor
from rgf2 import RGFClassifier
from stacked_generalization.lib.stacking import StackedClassifier, FWLSClassifier
from stacked_generalization.lib.util import saving_predict_proba, get_model_id
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


