
#.......................................
# skip a test if python is not available      [ see: https://github.com/rstudio/reticulate/tree/master/tests/testthat ]
#.......................................

skip_test_if_no_python <- function() {
  if (!reticulate::py_available(initialize = TRUE))
    testthat::skip("Python bindings not available for testing")
}


#.........................................
# skip a test if a module is not available      [ see: https://github.com/rstudio/reticulate ]
#.........................................

skip_test_if_no_module <- function(MODULE) {    # MODULE is of type character string ( length(MODULE) >= 1 )

  try({
    if (length(MODULE) == 1) {
      module_exists <- reticulate::py_module_available(MODULE)}
    else {
      module_exists <- sum(as.vector(sapply(MODULE, function(x) reticulate::py_module_available(x)))) == length(MODULE)
    }
  }, silent = TRUE)

  if (!module_exists) {
    testthat::skip(paste0(MODULE, " is not available for testthat-testing"))
  }
}
