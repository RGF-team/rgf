from distutils.dir_util import copy_tree
from platform import system
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.sdist import sdist
from shutil import rmtree
import io
import logging
import os
import subprocess
import sys


IS_64BITS = sys.maxsize > 2**32
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rgf_python')


def read(filename):
    return io.open(os.path.join(CURRENT_DIR, filename), encoding='utf-8').read()


def copy_files():

    def copy_files_helper(folder_name):
        src = os.path.join(CURRENT_DIR, os.path.pardir, folder_name)
        if os.path.isdir(src):
            dst = os.path.join(CURRENT_DIR, 'compile', folder_name)
            rmtree(dst, ignore_errors=True)
            copy_tree(src, dst, verbose=0)
        else:
            raise Exception('Cannot copy {} folder'.format(src))

    if not os.path.isfile(os.path.join(CURRENT_DIR, '_IS_SOURCE_PACKAGE')):
        copy_files_helper('RGF')
        copy_files_helper('FastRGF')


def clear_folder(path):
    if os.path.isdir(path):
        file_list = os.listdir(path)
        for file in file_list:
            try:
                file_path = os.path.abspath(os.path.join(path, file))
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    rmtree(file_path, ignore_errors=True)
            except Exception:
                pass


def find_rgf_lib():
    if system() in ('Windows', 'Microsoft'):
        exe_file = os.path.join(CURRENT_DIR, 'compile', 'RGF', 'bin', 'rgf.exe')
    else:
        exe_file = os.path.join(CURRENT_DIR, 'compile', 'RGF', 'bin', 'rgf')
    if os.path.isfile(exe_file):
        return exe_file
    else:
        return None


def find_fastrgf_lib():
    exe_files = []
    if system() in ('Windows', 'Microsoft'):
        exe_files.append(os.path.join(CURRENT_DIR, 'compile', 'FastRGF',
                                      'bin', 'forest_train.exe'))
        exe_files.append(os.path.join(CURRENT_DIR, 'compile', 'FastRGF',
                                      'bin', 'forest_predict.exe'))
    else:
        exe_files.append(os.path.join(CURRENT_DIR, 'compile', 'FastRGF',
                                      'bin', 'forest_train'))
        exe_files.append(os.path.join(CURRENT_DIR, 'compile', 'FastRGF',
                                      'bin', 'forest_predict'))
    for exe_file in exe_files:
        if not os.path.isfile(exe_file):
            return None
    return exe_files


def is_rgf_response(path):
    temp_x_loc = os.path.join(CURRENT_DIR, 'temp_rgf.train.data.x')
    temp_y_loc = os.path.join(CURRENT_DIR, 'temp_rgf.train.data.y')
    temp_model_loc = os.path.join(CURRENT_DIR, 'temp_rgf.model')
    temp_pred_loc = os.path.join(CURRENT_DIR, 'temp_rgf.predictions.txt')
    params_train = []
    params_train.append("train_x_fn=%s" % temp_x_loc)
    params_train.append("train_y_fn=%s" % temp_y_loc)
    params_train.append("model_fn_prefix=%s" % temp_model_loc)
    params_train.append("reg_L2=%s" % 1)
    params_train.append("max_leaf_forest=%s" % 10)
    params_pred = []
    params_pred.append("test_x_fn=%s" % temp_x_loc)
    params_pred.append("prediction_fn=%s" % temp_pred_loc)
    params_pred.append("model_fn=%s" % temp_model_loc + "-01")

    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass
    with open(temp_x_loc, 'w') as f:
        f.write('1 0 1 0\n0 1 0 1\n')
    with open(temp_y_loc, 'w') as f:
        f.write('1\n-1\n')
    success = silent_call((path, "train", ",".join(params_train)))
    success &= silent_call((path, "predict", ",".join(params_pred)))
    if success:
        return True
    else:
        return False


def is_fastrgf_response(path):
    temp_x_loc = os.path.join(CURRENT_DIR, 'temp_fastrgf.train.data.x')
    temp_y_loc = os.path.join(CURRENT_DIR, 'temp_fastrgf.train.data.y')
    temp_model_loc = os.path.join(CURRENT_DIR, "temp_fastrgf.model")
    temp_pred_loc = os.path.join(CURRENT_DIR, "temp_fastrgf.predictions.txt")
    path_train = os.path.join(path, ("forest_train.exe" if system() in ('Windows', 'Microsoft')
                                     else "forest_train"))
    params_train = []
    params_train.append("forest.ntrees=%s" % 10)
    params_train.append("tst.target=%s" % "BINARY")
    params_train.append("trn.x-file=%s" % temp_x_loc)
    params_train.append("trn.y-file=%s" % temp_y_loc)
    params_train.append("model.save=%s" % temp_model_loc)
    cmd_train = [path_train]
    cmd_train.extend(params_train)
    path_pred = os.path.join(path, ("forest_predict.exe" if system() in ('Windows', 'Microsoft')
                                    else "forest_predict"))
    params_pred = []
    params_pred.append("model.load=%s" % temp_model_loc)
    params_pred.append("tst.x-file=%s" % temp_x_loc)
    params_pred.append("tst.output-prediction=%s" % temp_pred_loc)
    cmd_pred = [path_pred]
    cmd_pred.extend(params_pred)
    try:
        os.chmod(path_train, os.stat(path_train).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.chmod(path_pred, os.stat(path_pred).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass
    with open(temp_x_loc, 'w') as X, open(temp_y_loc, 'w') as y:
        for _ in range(14):
            X.write('1 0 1 0\n0 1 0 1\n')
            y.write('1\n-1\n')
    success = silent_call(cmd_train)
    success &= silent_call(cmd_pred)
    if success:
        return True
    else:
        return False


def silent_call(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False


def compile_rgf():
    logger.info("Starting to compile RGF executable file.")
    success = False
    rgf_base_dir = os.path.join(CURRENT_DIR, 'compile', 'RGF')
    if not os.path.exists(os.path.join(rgf_base_dir, 'bin')):
        os.makedirs(os.path.join(rgf_base_dir, 'bin'))
    clear_folder(os.path.join(rgf_base_dir, 'bin'))  # Delete precompiled file
    if system() in ('Windows', 'Microsoft'):
        os.chdir(os.path.join(rgf_base_dir, 'Windows', 'rgf'))
        target = os.path.join(rgf_base_dir, 'bin', 'rgf.exe')
        logger.info("Trying to build executable file with MSBuild "
                    "from existing Visual Studio solution.")
        if IS_64BITS:
            arch = 'x64'
        else:
            arch = 'Win32'
        platform_toolsets = ('Windows7.1SDK', 'v100', 'v110',
                             'v120', 'v140', 'v141', 'v142')
        for platform_toolset in platform_toolsets:
            success = silent_call(('MSBuild',
                                   'rgf.sln',
                                   '/p:Configuration=Release',
                                   '/p:Platform={0}'.format(arch),
                                   '/p:PlatformToolset={0}'.format(platform_toolset)))
            clear_folder(os.path.join(rgf_base_dir, 'Windows', 'rgf', 'Release'))
            if success and os.path.isfile(target) and is_rgf_response(target):
                break
        os.chdir(os.path.join(rgf_base_dir, 'build'))
        if not success or not os.path.isfile(target) or not is_rgf_response(target):
            logger.warning("Building executable file with MSBuild "
                           "from existing Visual Studio solution failed.")
            logger.info("Trying to build executable file with MinGW g++ "
                        "from existing makefile.")
            success = silent_call(('mingw32-make'))
        if not success or not os.path.isfile(target) or not is_rgf_response(target):
            logger.warning("Building executable file with MinGW g++ "
                           "from existing makefile failed.")
            logger.info("Trying to build executable file with CMake and MSBuild.")
            generators = ('Visual Studio 10 2010', 'Visual Studio 11 2012',
                          'Visual Studio 12 2013', 'Visual Studio 14 2015',
                          'Visual Studio 15 2017', 'Visual Studio 16 2019')
            for generator in generators:
                clear_folder(os.path.join(rgf_base_dir, 'build'))
                success = silent_call(('cmake', '../', '-G', generator, '-A', arch))
                success &= silent_call(('cmake', '--build', '.', '--config', 'Release'))
                if success and os.path.isfile(target) and is_rgf_response(target):
                    break
        if not success or not os.path.isfile(target) or not is_rgf_response(target):
            logger.warning("Building executable file with CMake and MSBuild failed.")
            logger.info("Trying to build executable file with CMake and MinGW.")
            clear_folder(os.path.join(rgf_base_dir, 'build'))
            success = silent_call(('cmake', '../', '-G', 'MinGW Makefiles'))
            success &= silent_call(('cmake', '--build', '.', '--config', 'Release'))
    else:  # Linux, Darwin (macOS), etc.
        os.chdir(os.path.join(rgf_base_dir, 'build'))
        target = os.path.join(rgf_base_dir, 'bin', 'rgf')
        logger.info("Trying to build executable file with g++ from existing makefile.")
        success = silent_call(('make'))
        if not success or not os.path.isfile(target) or not is_rgf_response(target):
            logger.warning("Building executable file with g++ "
                           "from existing makefile failed.")
            logger.info("Trying to build executable file with CMake.")
            clear_folder(os.path.join(rgf_base_dir, 'build'))
            success = silent_call(('cmake', '../'))
            success &= silent_call(('cmake', '--build', '.', '--config', 'Release'))
    os.chdir(CURRENT_DIR)
    if success and os.path.isfile(target) and is_rgf_response(target):
        logger.info("Succeeded to build RGF.")
    else:
        logger.error("Compilation of RGF executable file failed. "
                     "Please build from binaries on your own and "
                     "specify path to the compiled file in the config file.")


def compile_fastrgf():

    def is_valid_gpp():
        tmp_result = False
        try:
            gpp_version = subprocess.check_output(('g++', '-dumpversion'),
                                                  universal_newlines=True,
                                                  stderr=subprocess.STDOUT)
            tmp_result = int(gpp_version[0]) >= 5
        except Exception:
            pass

        if tmp_result or system() in ('Windows', 'Microsoft'):
            return tmp_result

        for version in range(5, 10):
            try:
                subprocess.check_output(('g++-' + str(version), '--version'))
                return True
            except Exception:
                pass
        return tmp_result

    def is_valid_mingw():
        if not silent_call(('mingw32-make', '--version')):
            return False
        try:
            gpp_info = subprocess.check_output(('g++', '-v'),
                                               universal_newlines=True,
                                               stderr=subprocess.STDOUT)
            return gpp_info.find('posix') >= 0
        except Exception:
            return False

    logger.info("Starting to compile FastRGF executable files.")
    success = False
    fastrgf_base_dir = os.path.join(CURRENT_DIR, 'compile', 'FastRGF')
    if not os.path.exists(os.path.join(fastrgf_base_dir, 'bin')):
        os.makedirs(os.path.join(fastrgf_base_dir, 'bin'))
    if not os.path.exists(os.path.join(fastrgf_base_dir, 'build')):
        os.makedirs(os.path.join(fastrgf_base_dir, 'build'))
    if not silent_call(('cmake', '--version')):
        logger.error("Cannot compile FastRGF. "
                     "Make sure that you have installed CMake "
                     "and added path to it in environmental variable 'PATH'.")
        return
    if not is_valid_gpp():
        logger.error("Cannot compile FastRGF. "
                     "Compilation only with g++-5 and newer versions is possible.")
        return
    os.chdir(os.path.join(fastrgf_base_dir, 'build'))
    if system() in ('Windows', 'Microsoft'):
        if not is_valid_mingw():
            logger.error("Cannot compile FastRGF. "
                         "Make sure that you have installed MinGW-w64 "
                         "and added path to it in environmental variable 'PATH'.")
            os.chdir(CURRENT_DIR)
            return
        logger.info("Trying to build executable files with CMake and MinGW-w64.")
        target = os.path.join(fastrgf_base_dir, 'bin', 'forest_train.exe')
        success = silent_call(('cmake', '..', '-G', 'MinGW Makefiles'))
        success &= silent_call(('mingw32-make'))
        success &= silent_call(('mingw32-make', 'install'))
    else:  # Linux, Darwin (macOS), etc.
        logger.info("Trying to build executable files with CMake.")
        target = os.path.join(fastrgf_base_dir, 'bin', 'forest_train')
        success = silent_call(('cmake', '..'))
        success &= silent_call(('make'))
        success &= silent_call(('make', 'install'))
    os.chdir(CURRENT_DIR)
    if success and os.path.isfile(target) and is_fastrgf_response(os.path.dirname(target)):
        logger.info("Succeeded to build FastRGF.")
    else:
        logger.error("Compilation of FastRGF executable files failed. "
                     "Please build from binaries on your own and "
                     "specify path to the compiled files in the config file.")


class CustomSdist(sdist):

    def run(self):
        copy_files()
        tmp_flag_file_path = os.path.join(CURRENT_DIR, '_IS_SOURCE_PACKAGE')
        open(tmp_flag_file_path, 'w').close()
        sdist.run(self)
        if os.path.isfile(tmp_flag_file_path):
            os.remove(tmp_flag_file_path)


class CustomInstallLib(install_lib):
    def install(self):
        outfiles = install_lib.install(self)
        if not self.nocompilation:
            src = find_rgf_lib()
            if src:
                dst, _ = self.copy_file(src, os.path.join(self.install_dir, 'rgf'))
                outfiles.append(dst)
            else:
                logger.error("Cannot find RGF executable file. Installing without it.")
            sources = find_fastrgf_lib()
            if sources:
                for src in sources:
                    dst, _ = self.copy_file(src, os.path.join(self.install_dir, 'rgf'))
                    outfiles.append(dst)
            else:
                logger.error("Cannot find FastRGF executable files. Installing without them.")
        return outfiles


class CustomInstall(install):
    user_options = install.user_options \
                   + [('nocompilation', 'n', 'Install package without binaries.')]

    def initialize_options(self):
        install.initialize_options(self)
        self.nocompilation = False

    def run(self):
        if not self.nocompilation:
            logger.info("Starting to compile executable files.")
            copy_files()
            compile_rgf()
            compile_fastrgf()
        else:
            logger.info("Installing package without binaries.")
        install_lib = self.distribution.get_command_obj('install_lib')
        install_lib.user_options += [('nocompilation', 'n', 'Install package without binaries.')]
        install_lib.nocompilation = self.nocompilation
        install.run(self)


setup(name='rgf_python',
      version=read(os.path.join('rgf', 'VERSION')).strip(),
      description='Scikit-learn Wrapper for Regularized Greedy Forest',
      long_description=read('Readme.rst'),
      keywords='Machine Learning',
      author='RGF-team',
      author_email='nannyakannya@gmail.com',
      url='https://github.com/RGF-team/rgf/tree/master/python-package',
      license="MIT License",
      cmdclass={'install': CustomInstall,
                'install_lib': CustomInstallLib,
                'sdist': CustomSdist},
      packages=find_packages(),
      include_package_data=True,
      install_requires=["joblib", "six", "scikit-learn>=0.18"],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: MacOS',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'])
