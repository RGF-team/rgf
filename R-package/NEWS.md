## RGF 1.0.4

* We (RGF-team) modified the license from GPL-3 to MIT to go in accordance with the new structure of the *rgf_python* package. The package includes two files : *LICENSE* (for the RGF R package) and *LICENSE.note* (for the *RGF*, *FastRGF* and *rgf_python* packages).
* We added the following new features of RGF estimators : *feature_importances_* and *dump_model()*
* We modified the README.md file and especially the installation instructions for all operating systems (Linux, Macintosh, Windows)
* We created an R6 class (*Internal_class*) for all secondary functions which are used in RGF and FastRGF


## RGF 1.0.3

* The *dgCMatrix_2scipy_sparse* function was renamed to *TO_scipy_sparse* and now accepts either a *dgCMatrix* or a *dgRMatrix* as input. The appropriate format for the RGF package in case of sparse matrices is the *dgCMatrix* format (*scipy.sparse.csc_matrix*)
* We added an onload.R file to inform the users about the previous change
* Due to the previous changes we modified the Vignette and the tests too


## RGF 1.0.2

We commented the example(s) and test(s) related to the *dgCMatrix_2scipy_sparse* function [ *if (Sys.info()["sysname"] != 'Darwin')* ], because the *scipy-sparse* library on CRAN is not upgraded and the older version includes a bug (*TypeError : could not interpret data type*). This leads to an error on *Macintosh* ( *reference* : https://github.com/scipy/scipy/issues/5353 ).


## RGF 1.0.1

We added links to the github repository (master repository, issues).


## RGF 1.0.0

Initial version.