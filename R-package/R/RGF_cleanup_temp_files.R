#' Delete all temporary files of the created RGF estimators
#'
#' @details
#' This function deletes all temporary files of the created RGF estimators. See the issue \emph{https://github.com/RGF-team/rgf/issues/75} for more details.
#' @export
#' @references \emph{https://github.com/RGF-team/rgf/tree/master/python-package}
#' @examples
#'
#' \dontrun{
#' library(RGF)
#'
#' RGF_cleanup_temp_files()
#' }

RGF_cleanup_temp_files = function() {

  RGF_utils$cleanup()

  invisible()
}
