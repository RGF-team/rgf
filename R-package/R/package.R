#-------------------------
# Load the python-modules
#-------------------------


RGF_mod <- NULL; RGF_utils <- NULL; SCP <- NULL;


.onLoad <- function(libname, pkgname) {

  # print(reticulate::py_discover_config())
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

  reticulate::use_condaenv("test-environment", required = TRUE)

  if (reticulate::py_available(initialize = TRUE)) {

    if (reticulate::py_module_available("rgf.sklearn")) {

      RGF_mod <<- reticulate::import("rgf.sklearn", delay_load = TRUE)
    }
    else {
      packageStartupMessage("The 'rgf.sklearn' module is not available!")
    }

    if (reticulate::py_module_available("rgf.utils")) {

      RGF_utils <<- reticulate::import("rgf.utils", delay_load = TRUE)
    }
    else {
      packageStartupMessage("The 'rgf.utils' module is not available!")
    }

    if (reticulate::py_module_available("scipy")) {

      SCP <<- reticulate::import("scipy", delay_load = TRUE, convert = FALSE)
    }
    else {
      packageStartupMessage("The 'scipy' module is not available!")
    }
  }
  else {
    packageStartupMessage("Python is not available!")
  }
}
