
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/RGF)](http://cran.r-project.org/package=RGF)
[![Travis-CI Build Status](https://travis-ci.org/mlampros/RGF.svg?branch=master)](https://travis-ci.org/mlampros/RGF)
[![codecov.io](https://codecov.io/github/mlampros/RGF/coverage.svg?branch=master)](https://codecov.io/github/mlampros/RGF?branch=master)
[![Downloads](http://cranlogs.r-pkg.org/badges/grand-total/RGF?color=blue)](http://www.r-pkg.org/pkg/RGF)


## RGF (Regularized Greedy Forest)
<br>


The **RGF** package is a wrapper of the [Regularized Greedy Forest (RGF)](https://github.com/RGF-team/rgf_python) *python* package, which also includes a [Multi-core implementation (FastRGF)](https://github.com/baidu/fast_rgf). More details on the functionality of the RGF package can be found in the [blog-post](http://mlampros.github.io/2018/02/14/the_RGF_package/) and in the package Documentation. 

<br>

**UPDATE 26-07-2018**: A [Singularity image file](http://mlampros.github.io/2018/07/26/singularity_containers/) is available in case that someone intends to run *RGF* on Ubuntu Linux (locally or in a cloud instance) with all package requirements pre-installed. This allows the user to utilize the *RGF* package without having to spend time on the installation process.

<br>

**References:**

[Rie Johnson and Tong Zhang, Learning Nonlinear Functions Using Regularized Greedy Forest](https://arxiv.org/abs/1109.0887)

https://github.com/RGF-team/rgf/tree/master/RGF

https://github.com/RGF-team/rgf/tree/master/FastRGF

https://github.com/RGF-team/rgf/tree/master/python-package


<br>

### **System Requirements**

<br>

* Python (2.7 or >= 3.4)


<br>

All modules should be installed in the default python configuration (the configuration that the R-session displays as default), otherwise errors will occur during the *RGF* package installation (**reticulate::py_discover_config()** might be useful here). 

<br>

#### **Debian/Ubuntu/Fedora**    [ Python 2.7 ]

<br>

First install / upgrade the dependencies,

<br>

```R
sudo pip install --upgrade pip setuptools

sudo pip install -U numpy

sudo pip install --upgrade scipy

sudo pip install -U scikit-learn
```

<br>

Then, download the *rgf-python* package and install it using the following commands,

<br>


```R

git clone https://github.com/RGF-team/rgf.git

cd rgf/python-package

sudo python setup.py install

```

<br>

*FastRGF* will be installed successfully only if gcc >= 5.0.

<br><br>



#### **Macintosh OSX**            [ Python >= 3.4 ]

<br>

Upgrade python to version 3 using, 


```R

brew upgrade python

```
<br>

Then install the dependencies for *RGF* and *FastRGF*

<br>

```R
sudo pip3 install --upgrade setuptools

sudo pip3 install -U numpy

sudo pip3 install --upgrade scipy

sudo pip3 install -U scikit-learn
```
<br>


The *FastRGF* module requires a gcc >= 8.0. To install *gcc-8* (or the most recent gcc) with brew follow the next steps,

<br>

```R

# before running the following commands make sure that the most recent Apple command line tools for Xcode are installed

brew update

brew upgrade

brew info gcc

brew install gcc

brew cleanup

```
<br>

After the newest gcc version is installed the user should navigate to */usr/local/bin* and if a *gcc* file exists (symbolic link) then the user should delete it. Then the user should
run the following command,

<br>

```R

sudo ln -s /usr/local/bin/gcc-8 /usr/local/bin/gcc

```

<br>

The user should then verify that the gcc has been updated using,

<br>

```R

gcc -v

which gcc

```

<br>

After the new gcc is installed the user should continue with the installation of *rgf-python*,

<br>

```R

git clone https://github.com/RGF-team/rgf.git

cd /rgf/RGF/build

export CXX=/usr/local/bin/g++-8 && export CC=/usr/local/bin/gcc-8

cmake /rgf/RGF /rgf/FastRGF

make

sudo make install

cd /rgf/python-package

sudo python3 setup.py install

```
<br>

After a successful rgf-python installation the user should open an R session and give the following *reticulate* command to change to the relevant (brew-python) directory (otherwise the RGF package won't work properly),

<br>


```R

reticulate::use_python('/usr/local/bin/python3')


```

<br>

and then,

<br>


```R

reticulate::py_discover_config()


```

<br>

to validate that a user is in the python version where *RGF* or *FastRGF* are installed. Then,

<br>

```R

install.packages(RGF)


library(RGF)


```

<br>

to load the R package. It is possible that the following warning in the R session appears if *FastRGF* is not installed,

<br>

```R

UserWarning: Cannot find FastRGF executable files. FastRGF estimators will be unavailable for usage.
  warnings.warn("Cannot find FastRGF executable files. FastRGF estimators will be unavailable for usage.")
  
```

<br><br>

#### **Windows OS**            [ Python >= 3.4 ]

<br>

**NOTE : CURRENTLY THE PACKAGE ON WINDOWS CAN BE USED ONLY FROM THE COMMAND LINE (cmd)**

<br>

First download of [get-pip.py](https://bootstrap.pypa.io/get-pip.py) for windows

<br>

Update the Environment variables ( Control Panel >> System and Security >> System >> Advanced system settings >> Environment variables >> System variables >> Path >> Edit ) by adding ( for instance in case of python 3 ),

<br>

```R

C:\Python36;C:\Python36\Scripts


```

<br>

Install the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017)

<br>

Open the Command prompt (console) and install the *rgf_python* dependencies,

<br>

```R

pip3 install --upgrade setuptools

pip3 install -U numpy

pip3 install --upgrade scipy

pip3 install -U scikit-learn

```

<br>

Then download *git* for windows,

<br>

```R

https://git-scm.com/download/win


```
<br>

and run the downloaded  *.exe* file. Then do,

<br>

```R

git clone https://github.com/RGF-team/rgf.git

```

<br>

*FastRGF* requires a gcc version > 5.0 . To find out the gcc version, open a command prompt (console) and type,

<br>

```R

gcc --version


```

<br>

**Installation / Upgrade of MinGW**

<br>

**Perform the following steps to upgrade the MinGW (so that simple RGF functions work – but not FastRGF)**

<br>

Normally MinGW is installed in the **C:\\** directory. So, first delete the folder **C:\\MinGW** (if it already exists), and then remove the environment variable from (Control Panel >> System and Security >> System >> Advanced system settings >> Environment variables >> System variables >> Path >> Edit) which usually is **C:\\MinGW\\bin**. Then download the most recent version of [MinGW](http://www.mingw.org/wiki/Getting_Started), and especially the **mingw-get-setup.exe** which is an *automated GUI installer assistant*. After the new version is installed successfully, update the environment variable by adding **C:\\MinGW\\bin** in (Control Panel >> System and Security >> System >> Advanced system settings >> Environment variables >> System variables >> Path >> Edit). Then open a new command prompt (console) and type, 

<br>

```R

gcc --version


```
<br>

to find out if the new version of *MinGW* is installed properly.
 
<br>
 
A word of caution, If *Rtools* is already installed then make sure that it does not point to an older version of gcc. Just observe the *Path* field of the *environment variables* (accessible as explained previously).

<br>


**Perform the following steps only in case that a FastRGF installation is desired and gcc version is < 5.0**

<br>

*FastRGF* works only with [MinGW-w64](https://mingw-w64.org/doku.php ) because only this version provides POSIX threads. It can be downloaded from [MingW-W64-builds](https://mingw-w64.org/doku.php/download). After a successful download and installation the user should also update the environment variables field in (Control Panel >> System and Security >> System >> Advanced system settings >> Environment variables >> System variables >> Path >> Edit) by adding the following path (assuming the software is installed in **C:\\Program Files (x86)** folder),

<br>

```R

C:\Program Files (x86)\mingw-w64\i686-7.2.0-posix-dwarf-rt_v5-rev1\mingw32\bin


```

<br>

**Installation of cmake**

<br>

First download cmake for Windows, [win64-x64 Installer](https://cmake.org/download/).
Once the file is downloaded run the **.exe** file and during installation make sure to **add CMake to the system PATH for all users**.

<br>

Before the installation of *rgf* I might have to remove *Rtools* environment variables, such as **C:\\Rtools\\bin** (accessible as explained previously), otherwise errors might occur.

<br>

**Installation of RGF, FastRGF and rgf_python**   [ *assuming the installation takes place in the **c:/** directory* ]

<br>

Open a console with **administrator privileges** (right click on cmd and *run as administrator*), then 
<br>


```R


# download the most recent version of rgf-python from the github repository 
#--------------------------------------------------------------------------

git clone http://github.com/RGF-team/rgf.git



# then navigate to 
#-----------------

cd /rgf/RGF/

mkdir bin

cd c:/


# then download the latest "rgf.exe" from https://github.com/RGF-team/rgf/releases and place the "rgf.exe" inside the previously created "bin" folder ( /rgf/RGF/bin )



# installation of RGF
#--------------------

cd /rgf/RGF/build

mingw32-make

cd c:/



# installation of FastRGF
#------------------------

cd /rgf/FastRGF/

mkdir build

cd build


# BEFORE PROCEEDING WITH cmake MAKE SURE THAT THE "Rtools" folder IS NOT IN THE SAME DIRECTORY (IF THAT IS THE CASE THEN REMOVE IT TEMPROARILY, i.e. copy-paste the "Rtools" folder somewhere else)


cmake .. -G "MinGW Makefiles"

mingw32-make

mingw32-make install

cd c:/


# IF APPLICABLE, PASTE THE "Rtools" FOLDER IN THE INITIAL LOCATION / DIRECTORY


# installation of rgf-python
#---------------------------

cd rgf/python-package

python setup.py install


```

<br>

Then open a command prompt (console) and type,

<br>

```R

python 


```

<br>

to launch *Python* and then type 

<br>

```R

import rgf

exit()

```

<br>

to observe if *rgf* is installed properly. Then continue with the installation of the RGF package,

<br>

```R

install.packages(RGF)


```

<br>

On windows the user can take advantage of the RGF package currently **only** from within the command prompt (console). First, find the full path of the installation location of R (possible if someone right-clicks in the R short-cut (probably on Desktop) and navigates to properties >> shortcut >> target). In my OS, for instance, R is located in **C:\\Program Files\\R\\R-3.4.0\\bin\\x64\\R**. Then, by opening a command prompt (console) and giving (for instance in my case),

<br>

```R

cd C:\Program Files\R\R-3.4.0\bin\x64\

R

library(RGF)


```
<br>

one can proceed with the usage of the RGF package.

<br>


### **Installation of the RGF package**

<br>

To install the package from CRAN use, 

<br>

```R

install.packages('RGF')


```
<br>

and to download the latest version from Github use the *install_github* function of the devtools package,
<br><br>

```R

devtools::install_github(repo = 'mlampros/RGF')

```
<br>
Use the following link to report bugs/issues,
<br><br>

[https://github.com/mlampros/RGF/issues](https://github.com/mlampros/RGF/issues)

<br>

