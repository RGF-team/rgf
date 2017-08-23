|Build Status Travis| |Build Status AppVeyor| |License| |Python Versions| |PyPI Version|

.. [![PyPI Version](https://img.shields.io/pypi/v/rgf_python.svg)](https://pypi.python.org/pypi/rgf_python/) # Reserve link for PyPI in case of bugs at fury.io

rgf\_python
===========

The wrapper of machine learning algorithm **Regularized Greedy Forest (RGF)** `[1] <#reference>`__ for Python.

Features
--------

**Scikit-learn interface and possibility of usage for multiclass classification problem.**

Original RGF implementation is available only for regression and binary classification, but rgf\_python is also available for **multiclass classification** by "One-vs-Rest" method.

Example:

.. code:: python

    from sklearn import datasets
    from sklearn.utils.validation import check_random_state
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from rgf.sklearn import RGFClassifier

    iris = datasets.load_iris()
    rng = check_random_state(0)
    perm = rng.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]

    rgf = RGFClassifier(max_leaf=400,
                        algorithm="RGF_Sib",
                        test_interval=100,
                        verbose=True)

    n_folds = 3

    rgf_scores = cross_val_score(rgf,
                                 iris.data,
                                 iris.target,
                                 cv=StratifiedKFold(n_folds))

    rgf_score = sum(rgf_scores)/n_folds
    print('RGF Classfier score: {0:.5f}'.format(rgf_score))

More examples could be found `here <https://github.com/fukatani/rgf_python/tree/master/examples>`__.

Software Requirements
---------------------

-  Python (2.7 or >= 3.4)
-  scikit-learn (>= 0.18)

Installation
------------

From `PyPI <https://pypi.python.org/pypi/rgf_python>`__ using ``pip``:

::

    pip install rgf_python

or from `GitHub <https://github.com/fukatani/rgf_python>`__:

::

    git clone https://github.com/fukatani/rgf_python.git
    cd rgf_python
    python setup.py install

If you have any problems while installing by methods listed above you should *build RGF executable file from binaries by your own and place compiled executable file* into directory which is included in environmental variable **'PATH'** or into directory with installed package. Alternatively, you may specify actual location of RGF executable file and directory for placing temp files by corresponding flags in configuration file ``.rgfrc``, which you should create into your home directory. The default values are platform dependent: for Windows ``exe_location=$HOME/rgf.exe``, ``temp_location=$HOME/temp/rgf`` and for others ``exe_location=$HOME/rgf``, ``temp_location=/tmp/rgf``. Here is the example of ``.rgfrc`` file:

::

    exe_location=C:/Program Files/RGF/bin/rgf.exe
    temp_location=C:/Program Files/RGF/temp

Also, you may directly specify installation without automatic compilation:

::

    pip install rgf_python --install-option=--nocompilation

or

::

    git clone https://github.com/fukatani/rgf_python.git
    cd rgf_python
    python setup.py install --nocompilation

``sudo`` (or administrator privileges in Windows) may be needed to perform commands.

Here is the guide how you can build RGF executable file from binaries. The file will be in ``rgf_python/include/rgf/bin`` folder.

Windows
'''''''

Precompiled file
~~~~~~~~~~~~~~~~

The easiest way. Just take precompiled file from ``rgf_python/include/rgf/bin``.
For Windows 32-bit rename ``rgf32.exe`` to ``rgf.exe`` and take it.

Visual Studio (existing solution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open directory ``rgf_python/include/rgf/Windows/rgf``.
2. Open ``rgf.sln`` file with Visual Studio and choose ``BUILD->Build Solution (Ctrl+Shift+B)``.
   If you are asked to upgrade solution file after opening it click ``OK``.
   If you have errors about **Platform Toolset** go to ``PROJECT-> Properties-> Configuration Properties-> General`` and select the toolset installed on your machine.

MinGW (existing makefile)
~~~~~~~~~~~~~~~~~~~~~~~~~

Build executable file with MinGW g++ from existing ``makefile`` (you may want to customize this file for your environment).

::

    cd rgf_python/include/rgf/build
    mingw32-make

CMake and Visual Studio
~~~~~~~~~~~~~~~~~~~~~~~

Create solution file with CMake and then compile with Visual Studio.

::

    cd rgf_python/include/rgf/build
    cmake ../ -G "Visual Studio 10 2010"
    cmake --build . --config Release
    
If you are compiling on 64-bit machine then add ``Win64`` to the end of generator's name: ``Visual Studio 10 2010 Win64``. We tested following versions of Visual Studio:

- Visual Studio 10 2010 [Win64]
- Visual Studio 11 2012 [Win64]
- Visual Studio 12 2013 [Win64]
- Visual Studio 14 2015 [Win64]
- Visual Studio 15 2017 [Win64]

Other versions may work but are untested.

CMake and MinGW
~~~~~~~~~~~~~~~

Create ``makefile`` with CMake and then compile with MinGW.

::

    cd rgf_python/include/rgf/build
    cmake ../ -G "MinGW Makefiles"
    cmake --build . --config Release

\*nix
'''''

g++ (existing makefile)
~~~~~~~~~~~~~~~~~~~~~~~

Build executable file with g++ from existing ``makefile`` (you may want to customize this file for your environment).

::

    cd rgf_python/include/rgf/build
    make

CMake
~~~~~

Create ``makefile`` with CMake and then compile.

::

    cd rgf_python/include/rgf/build
    cmake ../
    cmake --build . --config Release

Tuning Hyper-parameters
-----------------------

You can tune hyper-parameters as follows.

-  *max\_leaf*: Appropriate values are data-dependent and usually varied from 1000 to 10000.
-  *test\_interval*: For efficiency, it must be either multiple or divisor of 100 (default value of the optimization interval).
-  *algorithm*: You can select "RGF", "RGF Opt" or "RGF Sib".
-  *loss*: You can select "LS", "Log" or "Expo".
-  *reg\_depth*: Must be no smaller than 1. Meant for being used with *algorithm* = "RGF Opt" or "RGF Sib".
-  *l2*: Either 1, 0.1, or 0.01 often produces good results though with exponential loss (*loss* = "Expo") and logistic loss (*loss* = "Log"), some data requires smaller values such as 1e-10 or 1e-20.
-  *sl2*: Default value is equal to *l2*. On some data, *l2*/100 works well.
-  *normalize*: If turned on, training targets are normalized so that the average becomes zero.
-  *min\_samples\_leaf*: Smaller values may slow down training. Too large values may degrade model accuracy.
-  *n\_iter*: Number of iterations of coordinate descent to optimize weights.
-  *n\_tree\_search*: Number of trees to be searched for the nodes to split. The most recently grown trees are searched first.
-  *opt\_interval*: Weight optimization interval in terms of the number of leaf nodes.
-  *learning\_rate*: Step size of Newton updates used in coordinate descent to optimize weights.

Detailed instruction of tuning hyper-parameters is `here <https://github.com/fukatani/rgf_python/blob/master/include/rgf/rgf1.2-guide.pdf>`__.

Using at Kaggle Kernel
----------------------

Now, Kaggle Kernel supports rgf\_python. Please see `this page <https://www.kaggle.com/fukatani/d/uciml/iris/classification-by-regularized-greedy-forest>`__.

License
-------

rgf_python is distributed under the GNU General Public License v3 (GPLv3). Please read file `LICENSE <https://github.com/fukatani/rgf_python/blob/master/LICENSE>`__ for more information.

rgf_python includes RGF version 1.2 which is distributed under the GPLv3. Original CLI implementation of RGF you can download at http://tongzhang-ml.org/software/rgf.

We thank Rie Johnson and Tong Zhang (authors of RGF).

Other
-----

Shamelessly, much part of the implementation is based on the following `code <https://github.com/MLWave/RGF-sklearn>`__. Thanks!

Reference
---------

[1] `Rie Johnson and Tong Zhang, Learning Nonlinear Functions Using Regularized Greedy Forest <https://arxiv.org/abs/1109.0887>`__ 

.. |Build Status Travis| image:: https://travis-ci.org/fukatani/rgf_python.svg?branch=master
   :target: https://travis-ci.org/fukatani/rgf_python
.. |Build Status AppVeyor| image:: https://ci.appveyor.com/api/projects/status/vpanb9hnncjr16hn/branch/master?svg=true
   :target: https://ci.appveyor.com/project/fukatani/rgf-python
.. |License| image:: https://img.shields.io/badge/license-GPLv3-blue.svg
   :target: https://github.com/fukatani/rgf_python/blob/master/LICENSE
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/rgf_python.svg
   :target: https://pypi.python.org/pypi/rgf_python/
.. |PyPI Version| image:: https://badge.fury.io/py/rgf_python.svg
   :target: https://badge.fury.io/py/rgf_python
