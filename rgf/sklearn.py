from __future__ import absolute_import

__all__ = ('RGFClassifier', 'RGFRegressor',
           'FastRGFClassifier', 'FastRGFRegressor')

import glob
from threading import Lock
import atexit
import codecs
import os
import platform
import subprocess

import numpy as np
from sklearn.externals import six
from rgf.rgf_model import RGFRegressor, RGFClassifier
from rgf.fastrgf_model import FastRGFRegressor, FastRGFClassifier


with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()


_NOT_FITTED_ERROR_DESC = "Estimator not fitted, call `fit` before exploiting the model."
_SYSTEM = platform.system()
_UUIDS = []
_FASTRGF_AVAILABLE = False


@atexit.register
def cleanup():
    for uuid in _UUIDS:
        cleanup_partial(uuid)


def cleanup_partial(uuid, remove_from_list=False):
    n_removed_files = 0
    if uuid in _UUIDS:
        model_glob = os.path.join(_TEMP_PATH, uuid + "*")
        for fn in glob.glob(model_glob):
            os.remove(fn)
            n_removed_files += 1
        if remove_from_list:
            _UUIDS.remove(uuid)
    return n_removed_files


def _get_paths():
    config = six.moves.configparser.RawConfigParser()
    path = os.path.join(os.path.expanduser('~'), '.rgfrc')

    try:
        with codecs.open(path, 'r', 'utf-8') as cfg:
            with six.StringIO(cfg.read()) as strIO:
                config.readfp(strIO)
    except six.moves.configparser.MissingSectionHeaderError:
        with codecs.open(path, 'r', 'utf-8') as cfg:
            with six.StringIO('[glob]\n' + cfg.read()) as strIO:
                config.readfp(strIO)
    except Exception:
        pass

    if _SYSTEM in ('Windows', 'Microsoft'):
        try:
            rgf_exe = os.path.abspath(config.get(config.sections()[0], 'exe_location'))
        except Exception:
            rgf_exe = os.path.join(os.path.expanduser('~'), 'rgf.exe')
        try:
            fast_rgf_path = os.path.abspath(config.get(config.sections()[0], 'fastrgf_location'))
        except Exception:
            fast_rgf_path = os.path.expanduser('~')
        try:
            temp = os.path.abspath(config.get(config.sections()[0], 'temp_location'))
        except Exception:
            temp = os.path.join(os.path.expanduser('~'), 'temp', 'rgf')
        def_exe = 'rgf.exe'
    else:  # Linux, Darwin (OS X), etc.
        try:
            rgf_exe = os.path.abspath(config.get(config.sections()[0], 'exe_location'))
        except Exception:
            rgf_exe = os.path.join(os.path.expanduser('~'), 'rgf')
        try:
            fast_rgf_path = os.path.abspath(config.get(config.sections()[0], 'fastrgf_location'))
        except Exception:
            fast_rgf_path = os.path.expanduser('~')
        try:
            temp = os.path.abspath(config.get(config.sections()[0], 'temp_location'))
        except Exception:
            temp = os.path.join('/tmp', 'rgf')
        def_exe = 'rgf'

    return def_exe, rgf_exe, fast_rgf_path, temp


_DEFAULT_EXE_PATH, _EXE_PATH, _FASTRGF_PATH, _TEMP_PATH = _get_paths()


if not os.path.isdir(_TEMP_PATH):
    os.makedirs(_TEMP_PATH)
if not os.access(_TEMP_PATH, os.W_OK):
    raise Exception("{0} is not writable directory. Please set "
                    "config flag 'temp_location' to writable directory".format(_TEMP_PATH))


def _is_rgf_executable(path):
    temp_x_loc = os.path.join(_TEMP_PATH, 'temp.train.data.x')
    temp_y_loc = os.path.join(_TEMP_PATH, 'temp.train.data.y')
    np.savetxt(temp_x_loc, [[1, 0, 1, 0], [0, 1, 0, 1]], delimiter=' ', fmt="%s")
    np.savetxt(temp_y_loc, [1, -1], delimiter=' ', fmt="%s")
    _UUIDS.append('temp')
    params = []
    params.append("train_x_fn=%s" % temp_x_loc)
    params.append("train_y_fn=%s" % temp_y_loc)
    params.append("model_fn_prefix=%s" % os.path.join(_TEMP_PATH, "temp.model"))
    params.append("reg_L2=%s" % 1)

    try:
        subprocess.check_output((path, "train", ",".join(params)))
        return True
    except Exception:
        return False


def _is_fastrgf_executable(path):
    train_exec = os.path.join(path, "forest_train")
    try:
        subprocess.check_output([train_exec, "--help"])
    except Exception:
        return False
    pred_exec = os.path.join(path, "forest_predict")
    try:
        subprocess.check_output([pred_exec, "--help"])
    except Exception:
        return False
    return True


if _is_rgf_executable(_DEFAULT_EXE_PATH):
    _EXE_PATH = _DEFAULT_EXE_PATH
elif _is_rgf_executable(os.path.join(os.path.dirname(__file__), _DEFAULT_EXE_PATH)):
    _EXE_PATH = os.path.join(os.path.dirname(__file__), _DEFAULT_EXE_PATH)
elif not os.path.isfile(_EXE_PATH):
    raise Exception("{0} is not executable file. Please set "
                    "config flag 'exe_location' to RGF execution file.".format(_EXE_PATH))
elif not os.access(_EXE_PATH, os.X_OK):
    raise Exception("{0} cannot be accessed. Please set "
                    "config flag 'exe_location' to RGF execution file.".format(_EXE_PATH))
elif _is_rgf_executable(_EXE_PATH):
    pass
else:
    raise Exception("{0} does not exist or {1} is not in the "
                    "'PATH' variable.".format(_EXE_PATH, _DEFAULT_EXE_PATH))

_FASTRGF_AVAILABLE = _is_fastrgf_executable(_FASTRGF_PATH)


def fastrgf_available():
    return _FASTRGF_AVAILABLE


def get_temp_path():
    return _TEMP_PATH


def get_exe_path():
    return _EXE_PATH


def get_fastrgf_path():
    return _FASTRGF_PATH


class _AtomicCounter(object):
    def __init__(self):
        self.value = 0
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value


_COUNTER = _AtomicCounter()
