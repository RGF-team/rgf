from platform import system
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.sdist import sdist
from sys import maxsize
import os
import subprocess


IS_64BITS = maxsize > 2**32
CURRENT_DIR = os.path.dirname(__file__)


def read(filename):
    return open(os.path.join(CURRENT_DIR, filename)).read()


def clear_folder(path):
    if os.path.isdir(path):
        file_list = os.listdir(path)
        for file in file_list:
            file_path = os.path.abspath(os.path.join(path, file))
            if os.path.isfile(file_path):
                os.remove(file_path)


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
    try:
        obj = subprocess.Popen((path, 'train', 'RGF_Sib'),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
        _ = obj.communicate()[0]
        if obj.returncode == 0:
            return True
        else:
            return False
    except Exception:
        return False


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
        print("Trying to build executable file with MSBuild "
              "from existing Visual Studio solution.")
        platform_toolsets = ('Windows7.1SDK', 'v100', 'v110', 'v120', 'v140',
                             'v141', 'v150')  # FIXME: Works only with W7.1SDK
        for platform_toolset in platform_toolsets:
            if IS_64BITS:
                arch = 'x64'
            else:
                arch = 'x86'
            status = os.system('MSBuild rgf.sln '
                               '/p:Configuration=Release '
                               '/p:Platform={0} '
                               '/p:PlatformToolset={1}'.format(arch, platform_toolset))
            clear_folder('Release')
            if status == 0 and os.path.isfile(target) and is_executable_response(target):
                break
        os.chdir(os.path.join(os.path.pardir, os.path.pardir, 'build'))
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
            print("Building executable file with MSBuild "
                  "from existing Visual Studio solution failed.")
            print("Trying to build executable file with MinGW g++ "
                  "from existing makefile.")
            status = os.system('mingw32-make')
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
            print("Building executable file with MinGW g++ "
                  "from existing makefile failed.")
            print("Trying to build executable file with CMake and MSBuild.")
            generators = ('Visual Studio 10 2010', 'Visual Studio 11 2012',
                          'Visual Studio 12 2013', 'Visual Studio 14 2015',
                          'Visual Studio 15 2017')
            for generator in generators:
                if IS_64BITS:
                    generator += ' Win64'
                clear_folder('.')
                status = os.system('cmake ../ -G "{0}"'.format(generator))
                status += os.system('cmake --build . --config Release')
                if status == 0 and os.path.isfile(target) and is_executable_response(target):
                    break
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
            print("Building executable file with CMake and MSBuild failed.")
            print("Trying to build executable file with CMake and MinGW g++.")
            clear_folder('.')
            status = os.system('cmake ../ -G "MinGW Makefiles"')
            status += os.system('cmake --build . --config Release')
        os.chdir(os.path.pardir)
    else:
        os.chdir('build')
        target = os.path.abspath(os.path.join(os.path.pardir, 'bin', 'rgf'))
        print("Trying to build executable file with g++ from existing makefile.")
        status = os.system('make')
        if status != 0 or not os.path.isfile(target) or not is_executable_response(target):
            print("Building executable file with g++ from existing makefile failed.")
            print("Trying to build executable file with CMake.")
            clear_folder('.')
            status = os.system('cmake ../')
            status += os.system('cmake --build . --config Release')
        os.chdir(os.path.pardir)
    os.chdir(os.path.join(os.path.pardir, os.path.pardir))
    if status:
        print('Error: Compilation of executable file failed. '
              'Please build from binaries by your own and '
              'specify path to the compiled file in the config file.')


class CustomInstallLib(install_lib):
    def install(self):
        outfiles = install_lib.install(self)
        src = find_lib()
        if src:
            dst, _ = self.copy_file(src, os.path.join(self.install_dir, 'rgf'))
            outfiles.append(dst)
        else:
            print('Error: Cannot find executable file. Installing without it.')
        return outfiles


class CustomInstall(install):
    user_options = install.user_options \
                   + [('nocompilation', 'n', 'Installing package without binaries.')]

    def initialize_options(self):
        install.initialize_options(self)
        self.nocompilation = 0

    def run(self):
        if not self.nocompilation:
            print('Starting to compile executable file.')
            compile_cpp()
        else:
            print('Installing package without binaries.')
        install.run(self)


setup(name='rgf_python',
      version=read(os.path.join('rgf', 'VERSION')).strip(),
      description='Scikit-learn Wrapper for Regularized Greedy Forest',
      long_description=read('Readme.rst'),
      keywords='Machine Learning',
      author='Ryosuke Fukatani',
      author_email='nannyakannya@gmail.com',
      url='https://github.com/fukatani/rgf_python',
      license="Apache License 2.0",
      cmdclass={'install': CustomInstall,
                'install_lib': CustomInstallLib},
      packages=find_packages(),
      include_package_data=True,
      install_requires=["scikit-learn>=0.18"],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
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
