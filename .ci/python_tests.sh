#!/bin/bash

if [[ $OS_NAME == "macos-latest" ]]; then
  brew install gcc
  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
else
  apt-get update
  apt-get install --no-install-recommends -y \
      apt-transport-https \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      gnupg-agent \
      software-properties-common
  if [[ $TASK != "R_PACKAGE" ]]; then
    add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt-get update
    apt-get install --no-install-recommends -y g++-5
    export CXX=g++-5 && export CC=gcc-5
  fi
  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
fi
bash miniconda.sh -b -p $CONDA_PATH
conda config --set always_yes yes --set changeps1 no
conda update -q conda
if [[ $TASK == "R_PACKAGE" ]]; then
  conda create -q -n $CONDA_ENV python=$PYTHON_VERSION pip openssl libffi --no-deps
  source activate $CONDA_ENV
  pip install setuptools joblib numpy scikit-learn scipy pandas wheel pytest
else
  conda create -q -n $CONDA_ENV python=$PYTHON_VERSION joblib numpy scikit-learn scipy pandas pytest
  source activate $CONDA_ENV
fi
cd $GITHUB_WORKSPACE/python-package
python setup.py sdist --formats gztar || exit -1
pip install dist/rgf_python-$RGF_VER.tar.gz -v || exit -1
#if [[ $TASK != "R_PACKAGE" ]]; then
pytest tests/ -v || exit -1
#fi
