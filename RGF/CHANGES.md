# March 2018 (version 1.3)

  1. Calculation of feature importance has been added.

  2. Dumping information about the forest model to the console has been added.

  3. License has been changed from GPLv3 to MIT.

**Breaking changes:**

  - Due to adding feature importance, old model files are not compatible with version `1.3`.

# February 2018

  1. Absolute error loss has been added.

# August-December 2017

  1. The executable file for 32-bit Windows has been added.

  2. Compilation with MinGW on Windows has been fixed.

  3. `CMakeLists.txt` file has been added.

  4. Compilation on 32-bit Windows has been fixed.
     Also, compilation with newer MS Visual Studios has been added.

# June 2014

This version (`1.2` with modifications listed below)
you can download [here](http://tongzhang-ml.org/software/rgf/index.html).

  1. The restriction on the size of training data files has been removed.
 
     _Old_: Training data files had to be smaller than 2GB.

     _New_: No restriction on the training data file sizes.
            (However, the number of lines and the length of each line must
            be smaller than 2^31 (2,147,483,648).)

  2. The solution file for MS Visual C++ 2010 Express has been changed
     from 32-bit to 64-bit; also, `__AZ_MSDN__` has been added to
     Preprocessor Definitions.

  3. Some source code files have been changed.

# September 2012 (version 1.2)

  1. The initial release.
