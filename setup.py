from platform import system
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from shutil import rmtree
from sys import maxsize
import io
import logging
import os
import subprocess


IS_64BITS = maxsize > 2**32
CURRENT_DIR = os.path.dirname(__file__)
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


def find_lib():
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
        with open(os.devnull, "w") as shut_up:
            subprocess.check_output(cmd, stderr=shut_up)
            return 0
    except Exception:
        return 1


def compile_cpp():
    status = 0
    os.chdir(os.path.join('include', 'rgf'))
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
            if status == 0 and os.path.isfile(target) and is_executable_response(target):
                break
        os.chdir(os.path.join(os.path.pardir, os.path.pardir, 'build'))
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
            logger.warning("Building executable file with MSBuild "
                           "from existing Visual Studio solution failed.")
            logger.info("Trying to build executable file with MinGW g++ "
                        "from existing makefile.")
            status = silent_call(('mingw32-make'))
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
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
                status += silent_call(('cmake', '--build', '.', '--config', 'Release'))
                if status == 0 and os.path.isfile(target) and is_executable_response(target):
                    break
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
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
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
            logger.warning("Building executable file with g++ "
                           "from existing makefile failed.")
            logger.info("Trying to build executable file with CMake.")
            clear_folder('.')
            status = silent_call(('cmake', '../'))
            status += silent_call(('cmake', '--build', '.', '--config', 'Release'))
        os.chdir(os.path.pardir)
    os.chdir(os.path.join(os.path.pardir, os.path.pardir))
    if status:
        logger.error("Compilation of executable file failed. "
                     "Please build from binaries by your own and "
                     "specify path to the compiled file in the config file.")


class CustomInstallLib(install_lib):
    def install(self):
        outfiles = install_lib.install(self)
        if not self.nocompilation:
            src = find_lib()
            if src:
                dst, _ = self.copy_file(src, os.path.join(self.install_dir, 'rgf'))
                outfiles.append(dst)
            else:
                logger.error("Cannot find executable file. Installing without it.")
        return outfiles


class CustomInstall(install):
    user_options = install.user_options \
                   + [('nocompilation', 'n', 'Installing package without binaries.')]

    def initialize_options(self):
        install.initialize_options(self)
        self.nocompilation = 0

    def run(self):
        if not self.nocompilation:
            logger.info("Starting to compile executable file.")
            compile_cpp()
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
