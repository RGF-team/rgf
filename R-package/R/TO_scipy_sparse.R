#' conversion of an R sparse matrix to a scipy sparse matrix
#'
#'
#' @param R_sparse_matrix an R sparse matrix. Acceptable input objects are either a \emph{dgCMatrix} or a \emph{dgRMatrix}.
#' @details
#' This function allows the user to convert either an R \emph{dgCMatrix} or a \emph{dgRMatrix} to a scipy sparse matrix (\emph{scipy.sparse.csc_matrix} or \emph{scipy.sparse.csr_matrix}). This is useful because the \emph{RGF} package accepts besides an R dense matrix also python sparse matrices as input.
#'
#' The \emph{dgCMatrix} class is a class of sparse numeric matrices in the compressed, sparse, \emph{column-oriented format}. The \emph{dgRMatrix} class is a class of sparse numeric matrices in the compressed, sparse, \emph{row-oriented format}.
#'
#' @export
#' @import reticulate
#' @importFrom Matrix Matrix
#' @references https://stat.ethz.ch/R-manual/R-devel/library/Matrix/html/dgCMatrix-class.html, https://stat.ethz.ch/R-manual/R-devel/library/Matrix/html/dgRMatrix-class.html, https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix
#' @examples
#'
#' try({
#'     if (reticulate::py_available(initialize = TRUE)) {
#'         if (reticulate::py_module_available("scipy")) {
#'
#'             if (Sys.info()["sysname"] != 'Darwin') {
#'
#'                 library(RGF)
#'
#'
#'                 # 'dgCMatrix' sparse matrix
#'                 #--------------------------
#'
#'                 data = c(1, 0, 2, 0, 0, 3, 4, 5, 6)
#'
#'                 dgcM = Matrix::Matrix(
#'                     data = data
#'                     , nrow = 3
#'                     , ncol = 3
#'                     , byrow = TRUE
#'                     , sparse = TRUE
#'                 )
#'
#'                 print(dim(dgcM))
#'
#'                 res = TO_scipy_sparse(dgcM)
#'
#'                 print(res$shape)
#'
#'
#'                 # 'dgRMatrix' sparse matrix
#'                 #--------------------------
#'
#'                 dgrM = as(dgcM, "RsparseMatrix")
#'
#'                 print(dim(dgrM))
#'
#'                 res_dgr = TO_scipy_sparse(dgrM)
#'
#'                 print(res_dgr$shape)
#'             }
#'         }
#'     }
#' }, silent = TRUE)

TO_scipy_sparse = function(R_sparse_matrix) {

    if (inherits(R_sparse_matrix, "dgCMatrix")) {
        py_obj <- SCP$sparse$csc_matrix(
            reticulate::tuple(
                R_sparse_matrix@x
                , R_sparse_matrix@i
                , R_sparse_matrix@p
            )
            , shape = reticulate::tuple(
                R_sparse_matrix@Dim[1]
                , R_sparse_matrix@Dim[2]
            )
        )
    }

    else if (inherits(R_sparse_matrix, "dgRMatrix")) {

        py_obj <- SCP$sparse$csr_matrix(
            reticulate::tuple(
                R_sparse_matrix@x
                , R_sparse_matrix@j
                , R_sparse_matrix@p
            )
            , shape = reticulate::tuple(
                R_sparse_matrix@Dim[1]
                , R_sparse_matrix@Dim[2]
            )
        )
    }

    else {
        stop("the 'R_sparse_matrix' parameter should be either a 'dgCMatrix' or a 'dgRMatrix' sparse matrix", call. = FALSE)
    }

    return(py_obj)
}
