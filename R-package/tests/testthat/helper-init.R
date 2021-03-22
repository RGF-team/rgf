
# prefer Python 3 if available  [ see: https://github.com/rstudio/reticulate/blob/master/tests/testthat/helper-init.R ]
if (!reticulate::py_available(initialize = FALSE) &&
    is.na(Sys.getenv("RETICULATE_PYTHON", unset = NA)))
{
  python <- Sys.which("python3")
  if (nzchar(python))
    reticulate::use_python(python, required = TRUE)
}
