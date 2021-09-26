## RGF 1.0.9
* We've added a 'packageStartupMessage' informing the user in case of the error 'attempt to apply non-function' that he/she has to use the 'reticulate::py_config()' before loading the package (in a new R session)


## RGF 1.0.8
* We've modified the DESCRIPTION file by adding the 'Orcid' Number for the person 'Lampros Mouselimis'
* We've removed the 'maintainer' in the DESCRIPTION because this field is created automatically
* We've removed the 'LazyData' in the DESCRIPTION file that gave a NOTE on the CRAN checks
* We've converted all 'reticulate::py_available(initialize = TRUE)' to 'reticulate::py_available(initialize = FALSE)' otherwise it would give a NOTE on the CRAN tests for the Windows Operating System
* We've removed all comments from the 'package.R' file
* We've added the 'inst' folder and the 'CITATION' file to cite the software and the original articles / software


## RGF 1.0.7
* We've modified the *package.R* file so that messages are printed to the console whenever Python or any of the required modules is not available. Moreover, for the R-package testing the conda environment parameter is adjusted ( this applies to the RGF-team Github repository and not to the CRAN package directly )
* We've modified the *.appveyor.yml* file to return the *artifacts* in order to observe if tests ran successfully ( this applies to the RGF-team Github repository and not to the CRAN package directly )
* We've added tests to increase the code coverage.
* We've dropped support for Python 2.7
* We've fixed also the invalid URL's in the README.md file
* We removed the 'zzz.R' file which included the message: 'Beginning from version 1.0.3 the 'dgCMatrix_2scipy_sparse' function was renamed to 'TO_scipy_sparse' and now accepts either a 'dgCMatrix' or a 'dgRMatrix' as input. The appropriate format for the 'RGF' package in case of sparse matrices is the 'dgCMatrix' format (scipy.sparse.csc_matrix)' as after 4 version updates is no longer required
* We've modified the '.onLoad' function in the 'package.R' file by removing 'reticulate::py_available(initialize = TRUE)' which forces reticulate to initialize Python and gives the following NOTE on CRAN 'Warning in system2(command = python, args = shQuote(config_script), stdout = TRUE,  : ..."' had status 2' (see: https://github.com/rstudio/reticulate/issues/730#issuecomment-594365528)


## RGF 1.0.6

* We've added the *init_model* parameter to the *RGFRegressor* and *RGFClassifier*
* We've added the *save_model* method to the *RGFRegressor* and *RGFClassifier*
* Source files were broken up into one file per exported object as of [#266](https://github.com/RGF-team/rgf/pull/266)
* Internal calls to estimator constructors were changed to use keyword, instead of positional, arguments. [#267](https://github.com/RGF-team/rgf/pull/267)


## RGF 1.0.5

The RGF R package was integrated in the home repository of the Regularized Greedy Forest (RGF) library (https://github.com/RGF-team).

* We downgraded the minimum required version of R to 3.2.0
* We modified / formatted the R files


## RGF 1.0.4

* We modified the license from GPL-3 to MIT to go in accordance with the new structure of the *rgf_python* package. The package includes two files : *LICENSE* (for the RGF R package) and *LICENSE.note* (for the *RGF*, *FastRGF* and *rgf_python* packages).
* We added the following new features of RGF estimators : *feature_importances_* and *dump_model()*
* We modified the README.md file and especially the installation instructions for all operating systems (Linux, Mac OS X, Windows)
* We created an R6 class (*Internal_class*) for all secondary functions which are used in RGF and FastRGF


## RGF 1.0.3

* The *dgCMatrix_2scipy_sparse* function was renamed to *TO_scipy_sparse* and now accepts either a *dgCMatrix* or a *dgRMatrix* as input. The appropriate format for the RGF package in case of sparse matrices is the *dgCMatrix* format (*scipy.sparse.csc_matrix*)
* We added an onload.R file to inform the users about the previous change
* Due to the previous changes we modified the Vignette and the tests too


## RGF 1.0.2

We commented the example(s) and test(s) related to the *dgCMatrix_2scipy_sparse* function [ *if (Sys.info()["sysname"] != 'Darwin')* ], because the *scipy-sparse* library on CRAN is not upgraded and the older version includes a bug (*TypeError : could not interpret data type*). This leads to an error on *Mac OS X* ( *reference* : https://github.com/scipy/scipy/issues/5353 ).


## RGF 1.0.1

We added links to the GitHub repository (master repository, issues).


## RGF 1.0.0

Initial version.
