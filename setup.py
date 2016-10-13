from setuptools import setup, find_packages

import re
import os

version = '0.0.1'

def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

try:
    import pypandoc
    read_md = lambda f: pypandoc.convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(name='rgf_sklearn',
      version=version,
      description='Scikit-Learn like Regularized Greedy Forest Wrapper',
      keywords = 'Machine Learning',
      author='Ryosuke Fukatani',
      author_email='nannyakannya@gmail.com',
      url='https://github.com/fukatani/rgf_sklearn',
      license="Apache License 2.0",
      packages=find_packages(),
      long_description=read_md('Readme.md'),
)

