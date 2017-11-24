FastRGF
=======

The wrapper of machine learning algorithm **FastRGF** `[1] <#reference>`__ for Python.

Features
--------

**Scikit-learn interface for FastRGF and possibility of usage for multiclass classification problem.**

FastRGF feature is alpha version.
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

Install rgf_python by `here <https://github.com/fukatani/rgf_python#installation>`__.

::

    git clone https://github.com/baidu/fast_rgf.git
    cd build/
    cmake ..
    make 
    make install

If you succeeded to make FastRGF, /path/to/fast_rgt/bin/forest_train and /path/to/fast_rgt/bin/forest_train should exist.
And you should indicate FastRGF location by ``~/.rgfrc`` file:

ex.
::

    exe_location=C:/Program Files/RGF/bin/rgf.exe
    temp_location=C:/Program Files/RGF/temp
    fastrgf_location=C:/Program Files/fast_rgf/bin

Reference
---------

[1] `Tong Zhang, FastRGF <https://github.com/baidu/fast_rgf>`__ 

