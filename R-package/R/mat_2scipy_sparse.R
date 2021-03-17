#' conversion of an R matrix to a scipy sparse matrix
#'
#'
#' @param x a data matrix
#' @param format a character string. Either \emph{"sparse_row_matrix"} or \emph{"sparse_column_matrix"}
#' @details
#' This function allows the user to convert an R matrix to a scipy sparse matrix. This is useful because the Regularized Greedy Forest algorithm accepts only python sparse matrices as input.
#' @export
#' @references https://docs.scipy.org/doc/scipy/reference/sparse.html
#' @examples
#'
#' try({
#'     if (reticulate::py_available(initialize = TRUE)) {
#'         if (reticulate::py_module_available("scipy")) {
#'
#'             library(RGF)
#'
#'             set.seed(1)
#'
#'             x = matrix(runif(1000), nrow = 100, ncol = 10)
#'
#'             res = mat_2scipy_sparse(x)
#'
#'             print(dim(x))
#'
#'             print(res$shape)
#'         }
#'     }
#' }, silent=TRUE)

mat_2scipy_sparse = function(x, format = 'sparse_row_matrix') {

    if (!inherits(x, "matrix")) {
        stop("the 'x' parameter should be of type 'matrix'", call. = FALSE)
    }

    if (format == 'sparse_column_matrix') {

        return(SCP$sparse$csc_matrix(x))

    } else if (format == 'sparse_row_matrix') {

        return(SCP$sparse$csr_matrix(x))

    } else {

        stop("the function can take either a 'sparse_row_matrix' or a 'sparse_column_matrix' for the 'format' parameter as input", call. = FALSE)
    }
}
