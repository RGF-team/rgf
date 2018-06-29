# FastRGF Version 0.6

### C++ Multi-core Implementation of Regularized Greedy Forest (RGF)

************************************************************************

# 0. Changes

Please see the file [`CHANGES.md`](./CHANGES.md) for the changelog of FastRGF.

************************************************************************

# Contents:

1. [Introduction](#1-introduction)

2. [Installation](#2-installation)

3. [Examples](#3-examples)

4. [Contact](#4-contact)

5. [Copyright](#5-copyright)

6. [References](#6-references)

# 1. Introduction

This software package provides a multi-core implementation of a simplified Regularized Greedy Forest (RGF) described in [[1]](#6-references). Please cite the paper if you find the software useful.

RGF is a machine learning method for building decision forests that have been used to win some Kaggle competitions. In our experience it works better than *gradient boosting* on many relatively large datasets.

The implementation employs the following concepts described in the original paper [[1]](#6-references):

-  tree node regularization;
-  fully-corrective update;
-  greedy node expansion with trade-off between leaf node splitting for current tree and root splitting for new tree.

However, various **simplifications** are made to accelerate the training speed. Therefore, unlike the [original RGF program](https://github.com/RGF-team/rgf/tree/master/RGF), this software does not reproduce the results in the paper.

The implementation of greedy tree node optimization employs second order Newton approximation for general loss functions. For logistic regression loss, which works especially well for many binary classification problems, this approach was considered in [[2]](#6-references); for general loss functions, second order approximation was considered in [[3]](#6-references).

# 2. Installation

The software is written in C++11 and requires to be built from the sources with the **g++** compiler. Note that compilation only with g++-5 and newer versions is possible. Other compilers are unsupported and older versions produce corrupted files.

[Download](https://github.com/RGF-team/rgf/archive/master.zip) the package and extract the content. Otherwise, you can use git

```
git clone https://github.com/RGF-team/rgf.git
```

The top directory of the extracted (or cloned) content is `rgf` and this software package is located into `FastRGF` subfolder. Below all the path expressions are relative to `rgf/FastRGF`.

The source files are located in the `include` and `src` directories.

The following executables will appear in the `bin` directory after you compile them by any method from the listed below.

- **forest_train**: train FastRGF and save the model;
- **forest_predict**: apply already trained model on test data.

You may use the option `-h` to show the command-line options of the each executable file.

## 2.1. Windows

On Windows compilation only with CMake and [MinGW-w64](https://mingw-w64.org/doku.php) is supported because only this version of MinGW provides POSIX threads.

```
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make
mingw32-make install
```

## 2.2. Unix-like Systems

```
mkdir build
cd build
cmake ..
make
make install
```

# 3. Examples

 Go to the [`examples`](./examples) subdirectory and follow the instructions in the [`README.md`](./examples/README.md) file. The file also contains some tips for parameter tuning.
 
# 4. Contact

Please post an [issue](https://github.com/RGF-team/rgf/issues) at GitHub repository for any errors you encounter.

# 5. Copyright

FastRGF is distributed under the **MIT license**. Please read the file [`LICENSE`](./LICENSE).

# 6. References

[1] [Rie Johnson and Tong Zhang. Learning Nonlinear Functions Using Regularized Greedy Forest.](https://arxiv.org/abs/1109.0887) IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(5):942-954, May 2014.

[2] [Ping Li. Robust LogitBoost and Adaptive Base Class (ABC) LogitBoost.](https://arxiv.org/abs/1203.3491) UAI, 2010.

[3] [Zhaohui Zheng, Hongyuan Zha, Tong Zhang, Olivier Chapelle, Keke Chen, Gordon Sun. A General Boosting Method and its Application to Learning Ranking Functions for Web Search.](https://papers.nips.cc/paper/3305-a-general-boosting-method-and-its-application-to-learning-ranking-functions-for-web-search) NIPS, 2007.
