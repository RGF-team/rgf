from setuptools import find_packages, setup
import os


def read(filenames):
    return open(os.path.join(os.path.dirname(__file__), filenames)).read()

try:
    import pypandoc
    read_md = lambda f: pypandoc.convert(f, 'rst')
except ImportError:
    print("Warning: pypandoc module not found, could not convert Markdown to RST.")
    read_md = lambda f: open(f, 'r').read()

setup(name='rgf_python',
      version=read(os.sep.join(['rgf', 'VERSION'])).strip(),
      description='Scikit-learn Wrapper for Regularized Greedy Forest',
      long_description=read_md('Readme.md'),
      keywords='Machine Learning',
      author='Ryosuke Fukatani',
      author_email='nannyakannya@gmail.com',
      url='https://github.com/fukatani/rgf_sklearn',
      license="Apache License 2.0",
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
