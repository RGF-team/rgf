from platform import system
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from shutil import rmtree
import sys
import io
import logging
import os
import subprocess


IS_64BITS = sys.maxsize > 2**32
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rgf_python')


def read(filename):
    return io.open(os.path.join(CURRENT_DIR, filename), encoding='utf-8').read()


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
    if system() in ('Windows',
                    'Microsoft') and os.path.isfile(os.path.join(CURRENT_DIR,
                                                                 'include',
                                                                 'rgf',
                                                                 'bin',
                                                                 'rgf.exe')):
        return os.path.join(CURRENT_DIR, 'include', 'rgf', 'bin', 'rgf.exe')
    elif os.path.isfile(os.path.join(CURRENT_DIR,
                                     'include',
                                     'rgf',
                                     'bin',
                                     'rgf')):
        return os.path.join(CURRENT_DIR, 'include', 'rgf', 'bin', 'rgf')
    else:
        return None


def find_fastrgf_lib():
    if system() in ('Windows', 'Microsoft'):
        return None
    elif os.path.isdir(os.path.join(CURRENT_DIR,
                                    'include/fast_rgf/build/src/exe')):
        return os.path.join(CURRENT_DIR, 'include/fast_rgf/build/src/exe')
    else:
        return None


def is_executable_response(path):
    temp_x_loc = os.path.abspath('temp.train.data.x')
    temp_y_loc = os.path.abspath('temp.train.data.y')
    params = []
    params.append("train_x_fn=%s" % temp_x_loc)
    params.append("train_y_fn=%s" % temp_y_loc)
    params.append("model_fn_prefix=%s" % os.path.abspath("temp.model"))
    params.append("reg_L2=%s" % 1)

    try:
        with open(temp_x_loc, 'w') as f:
            f.write('1 0 1 00 1 0 1\n')
        with open(temp_y_loc, 'w') as f:
            f.write('1-1\n')
        subprocess.check_output((path, "train", ",".join(params)))
        return True
    except Exception:
        return False


def silent_call(cmd):
    try:
        subprocess.check_output(cmd)
        return True
    except Exception:
        return False


def has_cmake_installed():
    return silent_call('cmake')


def has_mingw_make_installed():
    return silent_call('mingw32-make --version')


def compile_rgf():
    status = False
    os.chdir(os.path.join('include', 'rgf'))
    if not os.path.exists('bin'):
        os.makedirs('bin')
    clear_folder('bin')  # Delete precompiled file
    if system() in ('Windows', 'Microsoft'):
        os.chdir(os.path.join('Windows', 'rgf'))
        target = os.path.abspath(os.path.join(os.path.pardir,
                                              os.path.pardir,
                                              'bin',
                                              'rgf.exe'))
        logger.info("Trying to build executable file with MSBuild "
                    "from existing Visual Studio solution.")
        platform_toolsets = ('Windows7.1SDK', 'v100', 'v110',
                             'v120', 'v140', 'v141', 'v150')
        for platform_toolset in platform_toolsets:
            if IS_64BITS:
                arch = 'x64'
            else:
                arch = 'Win32'
            status = silent_call(('MSBuild',
                                  'rgf.sln',
                                  '/p:Configuration=Release',
                                  '/p:Platform={0}'.format(arch),
                                  '/p:PlatformToolset={0}'.format(platform_toolset)))
            clear_folder('Release')
            if status and os.path.isfile(target) and is_executable_response(target):
                break
        os.chdir(os.path.join(os.path.pardir, os.path.pardir, 'build'))
        if not status or not os.path.isfile(target) or not is_executable_response(target):
            logger.warning("Building executable file with MSBuild "
                           "from existing Visual Studio solution failed.")
            logger.info("Trying to build executable file with MinGW g++ "
                        "from existing makefile.")
            status = silent_call(('mingw32-make'))
        if not status or not os.path.isfile(target) or not is_executable_response(target):
            logger.warning("Building executable file with MinGW g++ "
                           "from existing makefile failed.")
            logger.info("Trying to build executable file with CMake and MSBuild.")
            generators = ('Visual Studio 10 2010', 'Visual Studio 11 2012',
                          'Visual Studio 12 2013', 'Visual Studio 14 2015',
                          'Visual Studio 15 2017')
            for generator in generators:
                if IS_64BITS:
                    generator += ' Win64'
                clear_folder('.')
                status = silent_call(('cmake', '../', '-G', generator))
                status &= silent_call(('cmake', '--build', '.', '--config', 'Release'))
                if not status and os.path.isfile(target) and is_executable_response(target):
                    break
        if not status or not os.path.isfile(target) or not is_executable_response(target):
            logger.warning("Building executable file with CMake and MSBuild failed.")
            logger.info("Trying to build executable file with CMake and MinGW.")
            clear_folder('.')
            status = silent_call(('cmake', '../', '-G', 'MinGW Makefiles'))
            status += silent_call(('cmake', '--build', '.', '--config', 'Release'))
        os.chdir(os.path.pardir)
    else:
        os.chdir('build')
        target = os.path.abspath(os.path.join(os.path.pardir, 'bin', 'rgf'))
        logger.info("Trying to build executable file with g++ from existing makefile.")
        status = silent_call(('make'))
        if not status or not os.path.isfile(target) or not is_executable_response(target):
            logger.warning("Building executable file with g++ "
                           "from existing makefile failed.")
            logger.info("Trying to build executable file with CMake.")
            clear_folder('.')
            status = silent_call(('cmake', '../'))
            status &= silent_call(('cmake', '--build', '.', '--config', 'Release'))
    os.chdir(CURRENT_DIR)
    if status:
        logger.info("Succeeded to build RGF.")
    else:
        logger.error("Compilation of rgf executable file failed. "
                     "Please build from binaries by your own and "
                     "specify path to the compiled file in the config file.")


def compile_fastrgf():

    def is_valid_gpp():
        for i in range(5, 8):
            try:
                subprocess.check_output(('g++-' + str(i), '--version'))
                return True
            except Exception:
                pass
        return False

    def is_valid_gpp_windows():
        try:
            result = subprocess.check_output(('g++', '--version'))
            if sys.version >= '3.0.0':
                result = result.decode()
            print(result)
            version = result.split('\n')[0].split(' ')[-1]
            return version >= '5.0.0'
        except Exception:
            pass
        return False

    if not has_cmake_installed():
        logger.info("FastRGF is not compiled because 'cmake' not found.")
        logger.info("If you want to use FastRGF, please compile yourself "
                    "after installed 'cmake'.")
        return
    if not os.path.exists('include/fast_rgf'):
        logger.info("Git submodule FastRGF is not found.")
        return
    if not os.path.isdir('include/fast_rgf/build'):
        os.mkdir('include/fast_rgf/build')
    os.chdir('include/fast_rgf/build')
    if system() in ('Windows', 'Microsoft'):
        if not has_mingw_make_installed():
            logger.info("FastRGF is not compiled because 'mingw32-make' not "
                        "found.")
            logger.info("If you want to use FastRGF, please compile yourself "
                        "after installed 'mingw32-make'.")
            return
        if not is_valid_gpp_windows():
            logger.info(
                "FastRGF is not compiled because FastRGF depends on g++>=5.0.0")
            return
        status = silent_call(('cmake', '..', '-G', '"MinGW Makefiles"'))
        status &= silent_call(('mingw32-make'))
        status &= silent_call(('mingw32-make', 'install'))
    else:
        if not is_valid_gpp():
            logger.info(
                "FastRGF is not compiled because FastRGF depends on g++>=5.0.0")
            return
        status = silent_call(('cmake', '..'))
        status &= silent_call(('make'))
        status &= silent_call(('make', 'install'))
    os.chdir(CURRENT_DIR)
    if status:
        logger.info("Succeeded to build FastRGF.")
    else:
        logger.error("Compilation of FastRGF executable file failed. "
                     "Please build from binaries by your own and "
                     "specify path to the compiled file in the config file.")


class CustomInstallLib(install_lib):
    def install(self):
        outfiles = install_lib.install(self)
        if not self.nocompilation:
            src = find_rgf_lib()
            if src:
                dst, _ = self.copy_file(src, os.path.join(self.install_dir, 'rgf'))
                outfiles.append(dst)
            else:
                logger.error("Cannot find rgf executable file. Installing without it.")
            src = find_fastrgf_lib()
            if src:
                forest_train = os.path.join(src, 'forest_train')
                dst, _ = self.copy_file(forest_train, os.path.join(self.install_dir, 'rgf'))
                outfiles.append(dst)
                forest_predict = os.path.join(src, 'forest_predict')
                dst, _ = self.copy_file(forest_predict, os.path.join(self.install_dir, 'rgf'))
                outfiles.append(dst)
            else:
                logger.error("Cannot find FastRGF executable file. Installing without it.")
        return outfiles


class CustomInstall(install):
    user_options = install.user_options \
                   + [('nocompilation', 'n', 'Installing package without binaries.')]

    def initialize_options(self):
        install.initialize_options(self)
        self.nocompilation = False

    def run(self):
        if not self.nocompilation:
            logger.info("Starting to compile executable file.")
            compile_rgf()
            compile_fastrgf()
        else:
            logger.info("Installing package without binaries.")
        install_lib = self.distribution.get_command_obj('install_lib')
        install_lib.user_options += [('nocompilation', 'n', 'Installing package without binaries.')]
        install_lib.nocompilation = self.nocompilation
        install.run(self)


setup(name='rgf_python',
      version=read(os.path.join('rgf', 'VERSION')).strip(),
      description='Scikit-learn Wrapper for Regularized Greedy Forest',
      long_description=read('Readme.rst'),
      keywords='Machine Learning',
      author='Ryosuke Fukatani',
      author_email='nannyakannya@gmail.com',
      url='https://github.com/fukatani/rgf_python',
      license="GNU General Public License v3 (GPLv3)",
      cmdclass={'install': CustomInstall,
                'install_lib': CustomInstallLib},
      packages=find_packages(),
      include_package_data=True,
      install_requires=["scikit-learn>=0.18"],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Natural Language :: English',
                   'Operating System :: MacOS',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'])
