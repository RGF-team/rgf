from __future__ import absolute_import

import os

from rgf.rgf_model import RGFRegressor, RGFClassifier
from rgf.fastrgf_model import FastRGFRegressor, FastRGFClassifier


__all__ = ('RGFClassifier', 'RGFRegressor',
           'FastRGFClassifier', 'FastRGFRegressor')


with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()
