from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.sdist import sdist
from platform import system
import os


_CURRENT_DIR = os.path.dirname(__file__)


def read(filename):
    return open(os.path.join(_CURRENT_DIR, filename)).read()


def find_lib():
    if system() in ('Windows',
                    'Microsoft') and os.path.isfile(os.path.join(_CURRENT_DIR,
                                                                 'include',
                                                                 'rgf',
                                                                 'bin',
                                                                 'rgf.exe')):
        return os.path.join(_CURRENT_DIR, 'include', 'rgf', 'bin', 'rgf.exe')
    elif os.path.isfile(os.path.join(_CURRENT_DIR,
                                     'include',
                                     'rgf',
                                     'bin',
                                     'rgf')):
        return os.path.join(_CURRENT_DIR, 'include', 'rgf', 'bin', 'rgf')
    else:
        return None

def compile_cpp():
    status = 0
    os.chdir(os.path.join('include', 'rgf'))
    if system() in ('Windows', 'Microsoft'):
        pass  # FIXME
#        if use_mingw:
#            cmake_cmd += " -G \"MinGW Makefiles\" "
#            os.system(cmake_cmd + " ../lightgbm/")
#            build_cmd = "mingw32-make.exe _lightgbm"
#        else:
#            vs_versions = ["Visual Studio 15 2017 Win64", "Visual Studio 14 2015 Win64", "Visual Studio 12 2013 Win64"]
#            try_vs = 1
#            for vs in vs_versions:
#                tmp_cmake_cmd = "%s -G \"%s\"" % (cmake_cmd, vs)
#                try_vs = os.system(tmp_cmake_cmd + " ../lightgbm/")
#                if try_vs == 0:
#                    cmake_cmd = tmp_cmake_cmd
#                    break
#                else:
#                    clear_path("./")
#            if try_vs != 0:
#                raise Exception('Please install Visual Studio or MS Build first')
#
#            build_cmd = "cmake --build . --target _lightgbm  --config Release"
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
            dst, _ = self.copy_file(src, os.path.join(self.install_dir,
                                                      os.path.split(src)[-1]))
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
