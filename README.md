[![Build Status Travis](https://travis-ci.org/RGF-team/rgf.svg?branch=master)](https://travis-ci.org/RGF-team/rgf)
[![Build Status AppVeyor](https://ci.appveyor.com/api/projects/status/u3612bfh9pmela42/branch/master?svg=true)](https://ci.appveyor.com/project/RGF-team/rgf)
[![DOI](https://zenodo.org/badge/DOI/10.1109/TPAMI.2013.159.svg)](https://doi.org/10.1109/TPAMI.2013.159)
[![arXiv.org](https://img.shields.io/badge/arXiv-1109.0887-b31b1b.svg)](https://arxiv.org/abs/1109.0887)
[![Python Versions](https://img.shields.io/pypi/pyversions/rgf_python.svg)](https://pypi.org/project/rgf_python)
[![PyPI Version](https://badge.fury.io/py/rgf_python.svg)](https://badge.fury.io/py/rgf_python)

# Regularized Greedy Forest

Regularized Greedy Forest (RGF) is a tree ensemble machine learning method described in [this paper](https://arxiv.org/abs/1109.0887).
RGF can deliver better results than gradient boosting decision tree (GBDT) on a number of datasets and it have been used to win some Kaggle competitions.
Unlike the traditional boosted decision tree approach, RGF directly works with the underlying forest structure.
RGF integrates two ideas: one is to include tree-structured regularization into the learning formulation; and the other is to employ the fully-corrective regularized greedy algorithm.

This repository contains the following implementations of the RGF algorithm:

- [RGF](https://github.com/RGF-team/rgf/tree/master/RGF): original implementation from the paper;
- [FastRGF](https://github.com/RGF-team/rgf/tree/master/FastRGF): multi-core implementation with some simplifications;
- [rgf_python](https://github.com/RGF-team/rgf/tree/master/python-package): wrapper of both RGF and FastRGF implementations for Python.

## Documentation
- [RGF](https://github.com/RGF-team/rgf/tree/master/RGF)
- [FastRGF](https://github.com/RGF-team/rgf/tree/master/FastRGF/README.md)
- [rgf_python](https://github.com/RGF-team/rgf/tree/master/python-package/Readme.rst)
- [RGF User Guide](https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.pdf)
- [Awesome RGF](https://github.com/RGF-team/rgf/tree/master/AWESOME_RGF.md)
