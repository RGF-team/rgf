#-------------------------
# Load the python-modules
#-------------------------


RGF_mod <- NULL; RGF_utils <- NULL; SCP <- NULL;


.onLoad <- function(libname, pkgname) {

  #---------------------------------------------------------------------------  keep these lines for debugging
  # print(reticulate::py_discover_config())               # it throws an error
  #
  # see the following issues / references:
  #  - https://github.com/rstudio/reticulate/issues/394
  #  - https://github.com/rstudio/reticulate/issues/292#issuecomment-399208184
  #  - https://github.com/rstudio/reticulate/issues/568#issuecomment-517926879
  #  - https://rdrr.io/github/kojikoji/ptcomp/src/R/zzz.R
  #  - http://r-pkgs.had.co.nz/r.html                       # usage of .onLoad
  #
  # available conda environments:
  # print(reticulate::conda_list(conda = "auto"))
  #               name                                                  python
  # 1  Miniconda36-x64                         C:\\Miniconda36-x64\\python.exe
  # 2 test-environment C:\\Miniconda36-x64\\envs\\test-environment\\python.exe
  # force reticulate to use the desired conda environment:
  # reticulate::use_condaenv('test-environment', required = TRUE)
  #---------------------------------------------------------------------------

  try({       # I added the try() functions in version 1.0.7 because I received a similar warning as mentioned in: [ https://github.com/rstudio/reticulate/issues/730#issuecomment-594365528 ] and [ https://github.com/rstudio/reticulate/issues/814 ]
    RGF_mod <<- reticulate::import("rgf.sklearn", delay_load = TRUE)
  }, silent=TRUE)

  try({
    RGF_utils <<- reticulate::import("rgf.utils", delay_load = TRUE)
  }, silent=TRUE)

  try({
    SCP <<- reticulate::import("scipy", delay_load = TRUE, convert = FALSE)
  }, silent=TRUE)

  # #................................................................................. keep this as a reference, however it gives a warning on CRAN because it tries to initialize python
  # try({
  #   if (reticulate::py_module_available("rgf.sklearn")) {
  #     RGF_mod <<- reticulate::import("rgf.sklearn", delay_load = TRUE)
  #   }
  #   # else {
  #   #   packageStartupMessage("The 'rgf.sklearn' module is not available!")          # keep these lines for debugging
  #   # }
  #   if (reticulate::py_module_available("rgf.utils")) {
  #     RGF_utils <<- reticulate::import("rgf.utils", delay_load = TRUE)
  #   }
  #   # else {
  #   #   packageStartupMessage("The 'rgf.utils' module is not available!")
  #   # }
  #   if (reticulate::py_module_available("scipy")) {
  #     SCP <<- reticulate::import("scipy", delay_load = TRUE, convert = FALSE)
  #   }
  #   # else {
  #   #   packageStartupMessage("The 'scipy' package is not available!")
  #   # }
  # }, silent=TRUE)
  # #.................................................................................
}
