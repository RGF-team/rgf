from platform import system
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.sdist import sdist
from sys import maxsize
import os


IS_64BITS = maxsize > 2**32
CURRENT_DIR = os.path.dirname(__file__)


def read(filename):
    return open(os.path.join(CURRENT_DIR, filename)).read()


def clear_folder(path):
    file_list = os.listdir(path)
    for file in file_list:
        file_path = os.path.join(path, file)
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


def compile_cpp():
    status = 0
    os.chdir(os.path.join('include', 'rgf'))
    if system() in ('Windows', 'Microsoft'):
        # Try to build with MSBuild
        os.chdir(os.path.join('Windows', 'rgf'))
        target = os.path.join(os.path.pardir,
                              os.pardir,
                              'bin',
                              'rgf.exe')
        platform_toolsets = ('v100', 'v110', 'v120', 'v140',
                             'v141', 'v150', 'v90', 'Windows7.1SDK')
        print("Trying to build executable file with MSBuild.")
        for platform_toolset in platform_toolsets:
            if IS_64BITS:
                status = os.system('MSBuild rgf.sln '
                                   '/p:ProjectConfiguration="Release|x64" '
                                   '/p:PlatformToolset={0}'.format(platform_toolset))
            else:
                status = os.system('MSBuild rgf.sln '
                                   '/p:ProjectConfiguration="Release|Win32" '
                                   '/p:PlatformToolset={0}'.format(platform_toolset))
            clear_folder('Release')
            if status == 0 and os.path.isfile(target):
                break
        if status != 0 or not os.path.isfile(target):
            # Try to build with MinGW
            print("Building executable file with MSBuild failed.")
            print("Trying to build executable file with MinGW.")
            # FIXME
        os.chdir(os.path.join(os.path.pardir, os.path.pardir))
    else:
        status = os.system('make')
    os.chdir(os.path.join(os.path.pardir, os.pardir))
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
