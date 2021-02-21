__all__ = ('RGFClassifier', 'RGFRegressor',
           'FastRGFClassifier', 'FastRGFRegressor')


import os

from rgf.rgf_model import RGFRegressor, RGFClassifier
from rgf.fastrgf_model import FastRGFRegressor, FastRGFClassifier


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'VERSION')) as _f:
    __version__ = _f.read().strip()
