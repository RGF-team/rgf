#-------------------------
# Load the python-modules
#-------------------------


RGF_mod <- NULL; RGF_utils <- NULL; SCP <- NULL;


.onLoad <- function(libname, pkgname) {

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
