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

4. [Hyperparameters Tuning](#4-hyperparameters-tuning)

5. [Known Issues](#5-known-issues)

6. [Contact](#6-contact)

7. [Copyright](#6-copyright)

8. [References](#7-references)

# 1. Introduction

This software package provides a multi-core implementation of a simplified Regularized Greedy Forest (RGF) described in [[1]](#7-references). Please cite the paper if you find the software useful.

RGF is a machine learning method for building decision forests that have been used to win some Kaggle competitions. In our experience it works better than *gradient boosting* on many relatively large datasets.

The implementation employs the following concepts described in the original paper [[1]](#7-references):

-  tree node regularization;
-  fully-corrective update;
-  greedy node expansion with trade-off between leaf node splitting for current tree and root splitting for new tree.

However, various **simplifications** are made to accelerate the training speed. Therefore, unlike the [original RGF program](https://github.com/RGF-team/rgf/tree/master/RGF), this software does not reproduce the results in the paper.

The implementation of greedy tree node optimization employs second order Newton approximation for general loss functions. For logistic regression loss, which works especially well for many binary classification problems, this approach was considered in [[2]](#7-references); for general loss functions, second order approximation was considered in [[3]](#7-references).

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

 Please go to the [`examples`](./examples) subdirectory.

# 4. Hyperparameters Tuning

## Forest-level hyperparameters

-  **forest.ntrees**: Controls the number of trees in the forest. Typical range is `[100, 10000]`. Default value is `500`.
-  **forest.opt**: Optimization method for training the forest. You can select `rgf` or `epsilon-greedy`. Default value is `rgf`.
-  **forest.stepsize**: Controls the step size of epsilon-greedy boosting. Meant for being used with `forest.opt=epsilon-greedy`. Default value is `0.0`.

## Tree-level hyperparameters

-  **dtree.max_level**: Controls the maximum tree depth. Default value is `6`.
-  **dtree.max_nodes**: Controls the maximum number of leaf nodes in best-first search. Default value is `50`.
-  **dtree.new_tree_gain_ratio**: Controls when to start a new tree. New tree is created when _leaf nodes gain < this value \* estimated gain of creating new tree_. Default value is `1.0`.
-  **dtree.min_sample**: Controls the minimum number of training data points in each leaf node. Default value is `5`.
-  **dtree.loss**: You can select `LS`, `MODLS` or `LOGISTIC` loss function. Default value is `LS`. However, for binary classification task `LOGISTIC` often works better.
-  **dtree.lamL1**: Controls the degree of L1 regularization. A large value induces sparsity. Typical range is `[0.0, 1000.0]`. Default value is `1.0`.
-  **dtree.lamL2**: Controls the degree of L2 regularization. The larger value is, the larger `forest.ntrees` you need to use: the resulting accuracy is often better with a longer training time. Use a relatively large value such as `1000.0` or `10000.0`. Default value is `1000.0`.

## Discretization hyperparameters

-  **discretize.sparse.max_buckets**: Controls the maximum number of discretized values. Typical range is `[10, 250]`. Default value is `200`. Meant for being used with sparse data.

   *If you want to try a larger value up to `65000`, then you need to edit [include/header.h](./include/header.h) and replace `using disc_sparse_value_t=unsigned char;` by `using disc_sparse_value_t=unsigned short;`. However, this will increase the memory usage.*
-  **discretize.dense.max_buckets**: Controls the maximum number of discretized values. Typical range is `[10, 65000]`. Default value is `65000`. Meant for being used with dense data.
-  **discretize.sparse.min_bucket_weights**: Controls the minimum number of effective samples for each discretized value. Default value is `5.0`. Meant for being used with sparse data.
-  **discretize.dense.min_bucket_weights**: Controls the minimum number of effective samples for each discretized value. Default value is `5.0`. Meant for being used with dense data.
-  **discretize.sparse.lamL2**: Controls the degree of L2 regularization for discretization. Default value is `2.0`. Meant for being used with sparse data.
-  **discretize.dense.lamL2**: Controls the degree of L2 regularization for discretization. Default value is `2.0`. Meant for being used with dense data.
-  **discretize.sparse.max_features**: Controls the maximum number of selected features. Typical range is `[1000, 10000000]`. Default value is `80000`. Meant for being used with sparse data.
-  **discretize.sparse.min_occurrences**: Controls the minimum number of occurrences for a feature to be selected. Default value is `5`. Meant for being used with sparse data.

# 5. Known Issues

- FastRGF crashes if training dataset is too small (#data < 28). [rgf#92](https://github.com/RGF-team/rgf/issues/92)
- FastRGF does not provide any built-in method to calculate feature importances. [rgf#109](https://github.com/RGF-team/rgf/issues/109)

# 6. Contact

Please post an [issue](https://github.com/RGF-team/rgf/issues) at GitHub repository for any errors you encounter.

# 7. Copyright

FastRGF is distributed under the **MIT license**. Please read the file [`LICENSE`](./LICENSE).

# 8. References

[1] Rie Johnson and Tong Zhang. [Learning Nonlinear Functions Using Regularized Greedy Forest.](https://arxiv.org/abs/1109.0887) IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(5):942-954, May 2014.

[2] Ping Li. [Robust LogitBoost and Adaptive Base Class (ABC) LogitBoost.](https://arxiv.org/abs/1203.3491) UAI, 2010.

[3] Zhaohui Zheng, Hongyuan Zha, Tong Zhang, Olivier Chapelle, Keke Chen, Gordon Sun. [A General Boosting Method and its Application to Learning Ranking Functions for Web Search.](https://papers.nips.cc/paper/3305-a-general-boosting-method-and-its-application-to-learning-ranking-functions-for-web-search) NIPS, 2007.
