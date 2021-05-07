#-------------------------
# Load the python-modules
#-------------------------


RGF_mod <- NULL; RGF_utils <- NULL; SCP <- NULL;


.onLoad <- function(libname, pkgname) {

  try({
    if (reticulate::py_available(initialize = FALSE)) {

      try({
        RGF_mod <<- reticulate::import("rgf.sklearn", delay_load = TRUE)
      }, silent = TRUE)

      try({
        RGF_utils <<- reticulate::import("rgf.utils", delay_load = TRUE)
      }, silent = TRUE)

      try({
        SCP <<- reticulate::import("scipy", delay_load = TRUE, convert = FALSE)
      }, silent = TRUE)

    }
  }, silent=TRUE)
}
