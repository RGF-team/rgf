FastRGF
=======

The wrapper of machine learning algorithm **FastRGF** `[1] <#reference>`__ for Python.

Features
--------

**Scikit-learn interface for FastRGF and possibility of usage for multiclass classification problem.**

FastRGF feature is **alpha version**.

The part of the function may be not tested, not documented and not unstabled. API can be changed in the future.

Example
-------

Examples could be found `here <https://github.com/fukatani/rgf_python/tree/master/examples>`__.

Software Requirements
---------------------

-  Python (2.7 or >= 3.4)
-  scikit-learn (>= 0.18)

Installation
------------

Install rgf_python according to `this guide <https://github.com/fukatani/rgf_python#installation>`__.

Then compile FastRGF.

Note that we test only g++-5 and newer compilators.

::

    git clone https://github.com/baidu/fast_rgf.git
    cd build/
    cmake ..
    make 
    make install

On Windows compilation only with `MinGW-w64 <https://mingw-w64.org/doku.php>`__ is supported because only this version provides POSIX threads:

::

    git clone https://github.com/baidu/fast_rgf.git
    cd build/
    cmake .. -G "MinGW Makefiles"
    mingw32-make 
    mingw32-make install

If you succeeded to make FastRGF, ``forest_train`` and ``forest_predict`` executable files should exist in ``fast_rgf/bin`` folder.
And you should indicate FastRGF location by ``~/.rgfrc`` file:

ex.

::

    exe_location=C:/Program Files/RGF/bin/rgf.exe
    temp_location=C:/Program Files/RGF/temp
    fastrgf_location=C:/Program Files/fast_rgf/bin

Reference
---------

[1] `Tong Zhang, FastRGF <https://github.com/baidu/fast_rgf>`__ 
