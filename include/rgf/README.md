# RGF Version 1.3

_Released: September 2012_

_Modified: March 2018_

## C++ programs for Regularized Greedy Forest (RGF)

************************************************************************

# 0.0 Changes

## 0.0.1 (March 2018)

  1. Calculating of feature importances has been added.

  2. Dumping information about the forest model to the console has been added.

  3. License has been changed from GPLv3 to MIT.

## 0.0.2 (February 2018)

  1. Absolute error loss has been added.

## 0.0.3 (August-December 2017)

  1. The executable file for 32-bit Windows has been added.

  2. `CMakeLists.txt` file has been added.

  3. Compilation on 32-bit Windows has been fixed.
     Also, compilation with newer MS Visual Studios has been added.

## 0.0.4 (June 2014)

  1. The restriction on the size of training data files has been removed.
 
     _Old_: Training data files had to be smaller than 2GB.

     _New_: No restriction on the training data file sizes.
            (However, the number of lines and the length of each line must
            be smaller than 2^31 (2,147,483,648).)

  2. The solution file for MS Visual C++ 2010 Express has been changed
     from 32-bit to 64-bit; also, `__AZ_MSDN__` has been added to
     Preprocessor Definitions.

  3. Some source code files have been changed for synchronization with
     my other software releases.

# 0.1 Acknowledgements

  We thank **Dave Slate** for suggesting a solution to the issue of file size
  restriction above.

************************************************************************

# Contents:

1. [Introduction](#1-introduction)

1.1. [System Requirements](#1-1-system-requirements)

2. [Download and Installation](#2-download-and-installation)

3. [Creating the Executable](#3-creating-the-executable)

3.1. [Windows](#3-1-windows)

3.2. [Unix-like Systems](#3-2-unix-like-systems)

3.3. [[Optional] Endianness Consideration](#3-3-optional-endianness-consideration)

4. [Documentation](#4-documentation)

5. [Contact](#5-contact)

6. [Copyright](#6-copyright)

7. [References](#7-references)

# 1. Introduction

This software package provides implementation of regularized greedy forest
(RGF) described in [[1]](#(7-references)).

## 1.1. System Requirements

Executables are provided only for some versions of Windows
(see [Section 3](#3-creating-the-executable)) for detail).
If provided executables do not work for your environment,
you need to compile C++ code.

To use the provided tools and to go through the examples in the user guide,
Perl is required.

# 2. Download and Installation

Download the package and extract the content.

The top directory of the extracted content is `rgf`. Below all the
path expressions are relative to `rgf`.

# 3. Creating the Executable

To go through the examples in the user guide, your executable needs to be
at the `bin` directory. Otherwise, your executable can be anywhere you like.

## 3.1. Windows

Executable files are provided with the filename `rgf.exe` for 64-bit Windows
and `rgf32.exe` for 32-bit Windows at the `bin` directory.
They were tested on Windows 10 with the latest updates as
of 8/12/2017. You can either use them
(rename `rgf32.exe` to `rgf.exe` in case you are using 32-bit Windows),
or you can rebuild them by yourself using the provided solution file for
MS Visual C++ 2010 Express: `Windows\rgf\rgf.sln`.

## 3.2. Unix-like Systems

You need to build your executable from the source code. A make file
`makefile` is provided at the `build` directory. It is configured to use
**g++** and always compile everything. You may need to customize `makefile`
for your environment.

To build the executable, change the current directory to the `build`
directory and enter in the command line `make`. Check the
`bin` directory to make sure that your new executable `rgf` is there.

## 3.3. [Optional] Endianness Consideration

The models obtained by RGF training can be saved to files.
The model files are essentially snap shots of memory that include
numerical values. Therefore, the model files are sensitive to
"endianness" of the environments. For this reason, if you wish to
share model files among environments of different endianness, you need
to follow the instructions below. Otherwise, you can skip this section.

To share model files among the environments of different endianness,
build your executable for the environment with big-endian with the
compile option:

```
/D_AZ_BIG_ENDIAN_
```

By doing so, the executable in your big-endian environment swaps the
byte order of numerical values before writing and after reading the
model files.

# 4. Documentation

`rgf-guide.pdf` "**Regularized Greedy Forest Version 1.2: User Guide**" is included.

# 5. Contact

riejohnson[at]gmail[dot]com

# 6. Copyright

RGF Version 1.3 is distributed under the MIT license. Please read
the file `COPYING`.

# 7. References

[1] Rie Johnson and Tong Zhang. Learning nonlinear functions using
    regularized greedy forest. IEEE Transactions on Pattern Analysis and Machine
    Intelligence, 36(5):942-954, May 2014.
