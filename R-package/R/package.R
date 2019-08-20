#-------------------------
# Load the python-modules
#-------------------------


RGF_mod <- NULL; RGF_utils <- NULL; SCP <- NULL;


.onLoad <- function(libname, pkgname) {

  # print(reticulate::py_discover_config())         # see this issue, probably related : https://github.com/rstudio/reticulate/issues/394
  # reticulate::use_condaenv("MINICONDA")
  reticulate::conda_list(conda = "auto")

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
