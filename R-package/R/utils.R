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
#' if (reticulate::py_available() && reticulate::py_module_available("scipy")) {
#'
#'   library(RGF)
#'
#'   set.seed(1)
#'
#'   x = matrix(runif(1000), nrow = 100, ncol = 10)
#'
#'   res = mat_2scipy_sparse(x)
#'
#'   print(dim(x))
#'
#'   print(res$shape)
#' }

mat_2scipy_sparse = function(x, format = 'sparse_row_matrix') {

  if (!inherits(x, "matrix")) {
      stop("the 'x' parameter should be of type 'matrix'", call. = FALSE)
  }

  if (format == 'sparse_column_matrix') {

    return(SCP$sparse$csc_matrix(x))

  } else if (format == 'sparse_row_matrix') {

    return(SCP$sparse$csr_matrix(x))

  } else {

    stop("the function can take either a 'sparse_row_matrix' or a 'sparse_column_matrix' for the 'format' parameter as input", call. = F)
  }
}


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
#' if (reticulate::py_available() && reticulate::py_module_available("scipy")) {
#'
#'   if (Sys.info()["sysname"] != 'Darwin') {
#'
#'     library(RGF)
#'
#'
#'     # 'dgCMatrix' sparse matrix
#'     #--------------------------
#'
#'     data = c(1, 0, 2, 0, 0, 3, 4, 5, 6)
#'
#'     dgcM = Matrix::Matrix(
#'         data = data
#'         , nrow = 3
#'         , ncol = 3
#'         , byrow = TRUE
#'         , sparse = TRUE
#'     )
#'
#'     print(dim(dgcM))
#'
#'     res = TO_scipy_sparse(dgcM)
#'
#'     print(res$shape)
#'
#'
#'     # 'dgRMatrix' sparse matrix
#'     #--------------------------
#'
#'     dgrM = as(dgcM, "RsparseMatrix")
#'
#'     print(dim(dgrM))
#'
#'     res_dgr = TO_scipy_sparse(dgrM)
#'
#'     print(res_dgr$shape)
#'   }
#' }
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

    stop("the 'R_sparse_matrix' parameter should be either a 'dgCMatrix' or a 'dgRMatrix' sparse matrix", call. = F)
  }

  return(py_obj)
}


#' Internal R6 class for all secondary functions used in RGF and FastRGF
#'
#' @importFrom R6 R6Class
#' @keywords internal
Internal_class <- R6::R6Class(
    "Internal_class",
    lock_objects = FALSE,
    public = list(

        # 'fit' function
        #----------------
        fit = function(x, y, sample_weight = NULL) {
            private$rgf_init$fit(x, y, sample_weight)
            return(invisible(NULL))
        },

        # 'predict' function
        #--------------------
        predict = function(x) {
            return(private$rgf_init$predict(x))
        },

        # 'predict' function [ probabilities ]
        #--------------------
        predict_proba = function(x) {
            return(private$rgf_init$predict_proba(x))
        },

        # 'cleanup' function
        #-------------------
        cleanup = function() {
            private$rgf_init$cleanup()
            return(invisible(NULL))
        },

        # 'get_params' function
        #----------------------
        get_params = function(deep = TRUE) {
            return(private$rgf_init$get_params(deep))
        },

        # score function
        #---------------
        score = function(x, y, sample_weight = NULL) {
            return(private$rgf_init$score(x, y, sample_weight))
        },

        # feature importance
        #-------------------
        feature_importances = function() {
            return(private$rgf_init$feature_importances_)
        },

        # dump-model
        #-----------
        dump_model = function() {
            return(private$rgf_init$dump_model)
        }

    ),

    private = list(
        rgf_init = NULL
    )
)


#' Regularized Greedy Forest regressor
#'
#'
#' @param x an R matrix (object) or a Python sparse matrix (object) of shape c(n_samples, n_features). The training input samples. The sparse matrix should be a Python sparse matrix. The helper functions \emph{mat_2scipy_sparse} and \emph{TO_scipy_sparse} allow the user to convert an R dense or sparse matrix to a scipy sparse matrix.
#' @param y a vector of shape c(n_samples). The target values (real numbers in regression).
#' @param sample_weight a vector of shape c(n_samples) or NULL. Individual weights for each sample.
#' @param max_leaf an integer. Training will be terminated when the number of leaf nodes in the forest reaches this value.
#' @param test_interval an integer. Test interval in terms of the number of leaf nodes.
#' @param algorithm a character string specifying the \emph{Regularization algorithm}. One of \emph{"RGF"} (RGF with L2 regularization on leaf-only models), \emph{"RGF_Opt"} (RGF with min-penalty regularization) or \emph{"RGF_Sib"} (RGF with min-penalty regularization with the sum-to-zero sibling constraints).
#' @param loss a character string specifying the \emph{Loss function}. One of \emph{"LS"} (Square loss), \emph{"Expo"} (Exponential loss) or \emph{"Log"} (Logistic loss).
#' @param reg_depth a float. Must be no smaller than 1.0. Meant for being used with the algorithm \emph{RGF Opt} or \emph{RGF Sib}. A larger value penalizes deeper nodes more severely.
#' @param l2 a float. Used to control the degree of L2 regularization.
#' @param sl2 a float or NULL. Override L2 regularization parameter l2 for the process of growing the forest. That is, if specified, the weight correction process uses l2 and the forest growing process uses sl2. If NULL, no override takes place and l2 is used throughout training.
#' @param normalize a boolean. If True, training targets are normalized so that the average becomes zero.
#' @param min_samples_leaf an integer or a float. Minimum number of training data points in each leaf node. If an integer, then consider \emph{min_samples_leaf} as the minimum number. If a float, then \emph{min_samples_leaf} is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
#' @param n_iter an integer or NULL. The number of iterations of coordinate descent to optimize weights. If NULL, 10 is used for loss = "LS" and 5 for loss = "Expo" or "Log".
#' @param n_tree_search an integer. The number of trees to be searched for the nodes to split. The most recently grown trees are searched first.
#' @param opt_interval an integer. Weight optimization interval in terms of the number of leaf nodes. For example, by default, weight optimization is performed every time approximately 100 leaf nodes are newly added to the forest.
#' @param learning_rate a float. Step size of Newton updates used in coordinate descent to optimize weights.
#' @param memory_policy a character string. One of \emph{"conservative"} (it uses less memory at the expense of longer runtime. Try only when with default value it uses too much memory) or \emph{"generous"} (it runs faster using more memory by keeping the sorted orders of the features on memory for reuse). Memory using policy.
#' @param verbose an integer. Controls the verbosity of the tree building process.
#' @export
#' @details
#'
#' the \emph{fit} function builds a regressor from the training set (x, y).
#'
#' the \emph{predict} function predicts the regression target for x.
#'
#' the \emph{cleanup} function removes tempfiles used by this model. See the issue \emph{https://github.com/RGF-team/rgf/issues/75}, which explains in which cases the \emph{cleanup} function applies.
#'
#' the \emph{get_params} function returns the parameters of the model.
#'
#' the \emph{score} function returns the coefficient of determination ( R^2 ) for the predictions.
#'
#' the \emph{feature_importances} function returns the feature importances for the data.
#'
#' the \emph{dump_model} function currently prints information about the fitted model in the console
#'
#' @references \emph{https://github.com/RGF-team/rgf/tree/master/python-package}, \emph{Rie Johnson and Tong Zhang, Learning Nonlinear Functions Using Regularized Greedy Forest}
#' @docType class
#' @importFrom R6 R6Class
#' @import reticulate
#' @section Methods:
#'
#' \describe{
#'  \item{\code{RGF_Regressor$new(max_leaf = 500, test_interval = 100,
#'                                algorithm = "RGF", loss = "LS", reg_depth = 1.0,
#'                                l2 = 0.1, sl2 = NULL, normalize = TRUE,
#'                                min_samples_leaf = 10, n_iter = NULL,
#'                                n_tree_search = 1, opt_interval = 100,
#'                                learning_rate = 0.5, memory_policy = "generous",
#'                                verbose = 0)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{fit(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{predict(x)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{cleanup()}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{get_params(deep = TRUE)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{score(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{feature_importances()}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{dump_model()}}{}
#'
#'  \item{\code{--------------}}{}
#'  }
#'
#' @usage # init <- RGF_Regressor$new(max_leaf = 500, test_interval = 100,
#' #                                  algorithm = "RGF", loss = "LS", reg_depth = 1.0,
#' #                                  l2 = 0.1, sl2 = NULL, normalize = TRUE,
#' #                                  min_samples_leaf = 10, n_iter = NULL,
#' #                                  n_tree_search = 1, opt_interval = 100,
#' #                                  learning_rate = 0.5, memory_policy = "generous",
#' #                                  verbose = 0)
#' @examples
#'
#' if (reticulate::py_available() && reticulate::py_module_available("rgf.sklearn")) {
#'
#'   library(RGF)
#'
#'   set.seed(1)
#'   x = matrix(runif(1000), nrow = 100, ncol = 10)
#'
#'   y = runif(100)
#'
#'   RGF_regr = RGF_Regressor$new(max_leaf = 50)
#'
#'   RGF_regr$fit(x, y)
#'
#'   preds = RGF_regr$predict(x)
#' }
RGF_Regressor <- R6::R6Class(
    "RGF_Regressor",
    inherit = Internal_class,
    lock_objects = FALSE,
    public = list(

        initialize = function(max_leaf = 500
                              , test_interval = 100
                              , algorithm = "RGF"
                              , loss = "LS"
                              , reg_depth = 1.0
                              , l2 = 0.1
                              , sl2 = NULL
                              , normalize = TRUE
                              , min_samples_leaf = 10
                              , n_iter = NULL
                              , n_tree_search = 1
                              , opt_interval = 100
                              , learning_rate = 0.5
                              , memory_policy = "generous"
                              , verbose = 0
        ){

            # exceptions for 'min_samples_leaf', 'n_iter'
            #--------------------------------------------

            # must be either > as.integer(1.0) or in (0, 0.5]
            if (min_samples_leaf >= 1.0) {
                min_samples_leaf <- as.integer(min_samples_leaf)
            }

            # must be either NULL or an integer
            if (!is.null(n_iter)) {
                n_iter <- as.integer(n_iter)
            }

            # initialize RGF_Regressor
            #------------------------
            private$rgf_init <- RGF_mod$RGFRegressor(
                as.integer(max_leaf)
                , as.integer(test_interval)
                , algorithm
                , loss
                , reg_depth
                , l2
                , sl2
                , normalize
                , min_samples_leaf
                , n_iter
                , as.integer(n_tree_search)
                , as.integer(opt_interval)
                , learning_rate
                , memory_policy
                , as.integer(verbose)
            )
        }
    )
)


#' Regularized Greedy Forest classifier
#'
#'
#' @param x an R matrix (object) or a Python sparse matrix (object) of shape c(n_samples, n_features). The training input samples. The sparse matrix should be a Python sparse matrix. The helper functions \emph{mat_2scipy_sparse} and \emph{TO_scipy_sparse} allow the user to convert an R dense or sparse matrix to a scipy sparse matrix.
#' @param y a vector of shape c(n_samples). The target values (class labels in classification).
#' @param sample_weight a vector of shape c(n_samples) or NULL. Individual weights for each sample.
#' @param max_leaf an integer. Training will be terminated when the number of leaf nodes in the forest reaches this value.
#' @param test_interval an integer. Test interval in terms of the number of leaf nodes.
#' @param algorithm a character string specifying the \emph{Regularization algorithm}. One of \emph{"RGF"} (RGF with L2 regularization on leaf-only models), \emph{"RGF_Opt"} (RGF with min-penalty regularization) or \emph{"RGF_Sib"} (RGF with min-penalty regularization with the sum-to-zero sibling constraints).
#' @param loss a character string specifying the \emph{Loss function}. One of \emph{"LS"} (Square loss), \emph{"Expo"} (Exponential loss) or \emph{"Log"} (Logistic loss).
#' @param reg_depth a float. Must be no smaller than 1.0. Meant for being used with the algorithm \emph{RGF Opt} or \emph{RGF Sib}. A larger value penalizes deeper nodes more severely.
#' @param l2 a float. Used to control the degree of L2 regularization.
#' @param sl2 a float or NULL. Override L2 regularization parameter l2 for the process of growing the forest. That is, if specified, the weight correction process uses l2 and the forest growing process uses sl2. If NULL, no override takes place and l2 is used throughout training.
#' @param normalize a boolean. If True, training targets are normalized so that the average becomes zero.
#' @param min_samples_leaf an integer or a float. Minimum number of training data points in each leaf node. If an integer, then consider \emph{min_samples_leaf} as the minimum number. If a float, then \emph{min_samples_leaf} is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
#' @param n_iter an integer or NULL. The number of iterations of coordinate descent to optimize weights. If NULL, 10 is used for loss = "LS" and 5 for loss = "Expo" or "Log".
#' @param n_tree_search an integer. The number of trees to be searched for the nodes to split. The most recently grown trees are searched first.
#' @param opt_interval an integer. Weight optimization interval in terms of the number of leaf nodes. For example, by default, weight optimization is performed every time approximately 100 leaf nodes are newly added to the forest.
#' @param learning_rate a float. Step size of Newton updates used in coordinate descent to optimize weights.
#' @param calc_prob a character string. One of \emph{"sigmoid"} or \emph{"softmax"}. Method of probability calculation.
#' @param n_jobs an integer. The number of jobs (threads) to use for the computation. The substantial number of the jobs dependents on \emph{classes_} (The number of classes when \emph{fit} is performed). If classes_ = 2, the substantial max number of the jobs is one. If classes_ > 2, the substantial max number of the jobs is the same as classes_. If n_jobs = 1, no parallel computing code is used at all regardless of classes_. If n_jobs = -1 and classes_ >= number of CPU, all CPUs are used. For n_jobs = -2, all CPUs but one are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
#' @param memory_policy a character string. One of \emph{"conservative"} (it uses less memory at the expense of longer runtime. Try only when with default value it uses too much memory) or \emph{"generous"} (it runs faster using more memory by keeping the sorted orders of the features on memory for reuse). Memory using policy.
#' @param verbose an integer. Controls the verbosity of the tree building process.
#' @export
#' @details
#'
#' the \emph{fit} function builds a classifier from the training set (x, y).
#'
#' the \emph{predict} function predicts the class for x.
#'
#' the \emph{predict_proba} function predicts class probabilities for x.
#'
#' the \emph{cleanup} function removes tempfiles used by this model. See the issue \emph{https://github.com/RGF-team/rgf/issues/75}, which explains in which cases the \emph{cleanup} function applies.
#'
#' the \emph{get_params} function returns the parameters of the model.
#'
#' the \emph{score} function returns the mean accuracy on the given test data and labels.
#'
#' the \emph{feature_importances} function returns the feature importances for the data.
#'
#' the \emph{dump_model} function currently prints information about the fitted model in the console
#'
#' @references \emph{https://github.com/RGF-team/rgf/tree/master/python-package}, \emph{Rie Johnson and Tong Zhang, Learning Nonlinear Functions Using Regularized Greedy Forest}
#' @docType class
#' @importFrom R6 R6Class
#' @import reticulate
#' @section Methods:
#'
#' \describe{
#'  \item{\code{RGF_Classifier$new(max_leaf = 1000, test_interval = 100,
#'                                algorithm = "RGF", loss = "Log", reg_depth = 1.0,
#'                                l2 = 0.1, sl2 = NULL, normalize = FALSE,
#'                                min_samples_leaf = 10, n_iter = NULL,
#'                                n_tree_search = 1, opt_interval = 100,
#'                                learning_rate = 0.5, calc_prob = "sigmoid",
#'                                n_jobs = 1, memory_policy = "generous",
#'                                verbose = 0)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{fit(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{predict(x)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{predict_proba(x)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{cleanup()}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{get_params(deep = TRUE)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{score(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{feature_importances()}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{dump_model()}}{}
#'
#'  \item{\code{--------------}}{}
#'  }
#'
#' @usage # init <- RGF_Classifier$new(max_leaf = 1000, test_interval = 100,
#' #                                  algorithm = "RGF", loss = "Log", reg_depth = 1.0,
#' #                                  l2 = 0.1, sl2 = NULL, normalize = FALSE,
#' #                                  min_samples_leaf = 10, n_iter = NULL,
#' #                                  n_tree_search = 1, opt_interval = 100,
#' #                                  learning_rate = 0.5, calc_prob = "sigmoid",
#' #                                  n_jobs = 1, memory_policy = "generous",
#' #                                  verbose = 0)
#' @examples
#'
#' if (reticulate::py_available() && reticulate::py_module_available("rgf.sklearn")) {
#'
#'   library(RGF)
#'
#'   set.seed(1)
#'   x = matrix(runif(1000), nrow = 100, ncol = 10)
#'
#'   y = sample(1:2, 100, replace = TRUE)
#'
#'   RGF_class = RGF_Classifier$new(max_leaf = 50)
#'
#'   RGF_class$fit(x, y)
#'
#'   preds = RGF_class$predict_proba(x)
#' }

RGF_Classifier <- R6::R6Class(
    "RGF_Classifier",
    inherit = Internal_class,
    lock_objects = FALSE,
    public = list(

        initialize = function(max_leaf = 1000
                              , test_interval = 100
                              , algorithm = "RGF"
                              , loss = "Log"
                              , reg_depth = 1.0
                              , l2 = 0.1
                              , sl2 = NULL
                              , normalize = FALSE
                              , min_samples_leaf = 10
                              , n_iter = NULL
                              , n_tree_search = 1
                              , opt_interval = 100
                              , learning_rate = 0.5
                              , calc_prob = "sigmoid"
                              , n_jobs = 1
                              , memory_policy = "generous"
                              , verbose = 0
        ) {

            # exceptions for 'min_samples_leaf', 'n_iter'
            #--------------------------------------------

            # must be either > as.integer(1.0) or in (0, 0.5]
            if (min_samples_leaf >= 1.0) {
                min_samples_leaf <- as.integer(min_samples_leaf)
            }

            # must be either NULL or an integer
            if (!is.null(n_iter)) {
                n_iter <- as.integer(n_iter)
            }

            # initialize RGF_Classifier
            #--------------------------
            private$rgf_init <- RGF_mod$RGFClassifier(
                as.integer(max_leaf)
                , as.integer(test_interval)
                , algorithm
                , loss
                , reg_depth
                , l2
                , sl2
                , normalize
                , min_samples_leaf
                , n_iter
                , as.integer(n_tree_search)
                , as.integer(opt_interval)
                , learning_rate
                , calc_prob
                , as.integer(n_jobs)
                , memory_policy
                , as.integer(verbose)
            )
        }
    )
)


#' A Fast Regularized Greedy Forest regressor
#'
#'
#' @param x an R matrix (object) or a Python sparse matrix (object) of shape c(n_samples, n_features). The training input samples. The sparse matrix should be a Python sparse matrix. The helper functions \emph{mat_2scipy_sparse} and \emph{TO_scipy_sparse} allow the user to convert an R dense or sparse matrix to a scipy sparse matrix.
#' @param y a vector of shape c(n_samples). The target values (real numbers in regression).
#' @param n_estimators an integer. The number of trees in the forest (Original name: forest.ntrees.)
#' @param max_depth an integer. Maximum tree depth (Original name: dtree.max_level.)
#' @param max_leaf an integer. Maximum number of leaf nodes in best-first search (Original name: dtree.max_nodes.)
#' @param tree_gain_ratio a float. New tree is created when leaf-nodes gain < this value * estimated gain of creating new tree (Original name: dtree.new_tree_gain_ratio.)
#' @param min_samples_leaf an integer or float. Minimum number of training data points in each leaf node. If an integer, then consider min_samples_leaf as the minimum number. If a float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node (Original name: dtree.min_sample.)
#' @param l1 a float. Used to control the degree of L1 regularization (Original name: dtree.lamL1.)
#' @param l2 a float. Used to control the degree of L2 regularization (Original name: dtree.lamL2.)
#' @param opt_algorithm a character string. Either \emph{"rgf"} or \emph{"epsilon-greedy"}. Optimization method for training forest (Original name: forest.opt.)
#' @param learning_rate a float. Step size of epsilon-greedy boosting. Meant for being used with opt_algorithm = "epsilon-greedy" (Original name: forest.stepsize.)
#' @param max_bin an integer or NULL. Maximum number of discretized values (bins). If NULL, 65000 is used for dense data and 200 for sparse data (Original name: discretize.(sparse/dense).max_buckets.)
#' @param min_child_weight a float. Minimum sum of data weights for each discretized value (bin) (Original name: discretize.(sparse/dense).min_bucket_weights.)
#' @param data_l2 a float. Used to control the degree of L2 regularization for discretization (Original name: discretize.(sparse/dense).lamL2.)
#' @param sparse_max_features an integer. Maximum number of selected features. Meant for being used with sparse data (Original name: discretize.sparse.max_features.)
#' @param sparse_min_occurences an integer. Minimum number of occurrences for a feature to be selected. Meant for being used with sparse data (Original name: discretize.sparse.min_occrrences.)
#' @param n_jobs an integer. The number of jobs to run in parallel for both fit and predict. If -1, all CPUs are used. If -2, all CPUs but one are used. If < -1, (n_cpus + 1 + n_jobs) are used (Original name: set.nthreads.)
#' @param verbose an integer. Controls the verbosity of the tree building process (Original name: set.verbose.)
#' @export
#' @details
#'
#' the \emph{fit} function builds a regressor from the training set (x, y).
#'
#' the \emph{predict} function predicts the regression target for x.
#'
#' the \emph{cleanup} function removes tempfiles used by this model. See the issue \emph{https://github.com/RGF-team/rgf/issues/75}, which explains in which cases the \emph{cleanup} function applies.
#'
#' the \emph{get_params} function returns the parameters of the model.
#'
#' the \emph{score} function returns the coefficient of determination ( R^2 ) for the predictions.
#'
#' @references \emph{https://github.com/RGF-team/rgf/tree/master/python-package}, \emph{Tong Zhang, FastRGF: Multi-core Implementation of Regularized Greedy Forest (https://github.com/RGF-team/rgf/tree/master/FastRGF)}
#' @docType class
#' @importFrom R6 R6Class
#' @import reticulate
#' @section Methods:
#'
#' \describe{
#'  \item{\code{FastRGF_Regressor$new(n_estimators = 500, max_depth = 6,
#'                                    max_leaf = 50, tree_gain_ratio = 1.0,
#'                                    min_samples_leaf = 5, l1 = 1.0,
#'                                    l2 = 1000.0, opt_algorithm = "rgf",
#'                                    learning_rate = 0.001, max_bin = NULL,
#'                                    min_child_weight = 5.0, data_l2 = 2.0,
#'                                    sparse_max_features = 80000,
#'                                    sparse_min_occurences = 5,
#'                                    n_jobs = 1, verbose = 0)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{fit(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{predict(x)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{cleanup()}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{get_params(deep = TRUE)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{score(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'  }
#'
#' @usage # init <- FastRGF_Regressor$new(n_estimators = 500, max_depth = 6,
#' #                                      max_leaf = 50, tree_gain_ratio = 1.0,
#' #                                      min_samples_leaf = 5, l1 = 1.0,
#' #                                      l2 = 1000.0, opt_algorithm = "rgf",
#' #                                      learning_rate = 0.001, max_bin = NULL,
#' #                                      min_child_weight = 5.0, data_l2 = 2.0,
#' #                                      sparse_max_features = 80000,
#' #                                      sparse_min_occurences = 5,
#' #                                      n_jobs = 1, verbose = 0)
#' @examples
#'
#' if (reticulate::py_available() && reticulate::py_module_available("rgf.sklearn")) {
#'
#'   library(RGF)
#'
#'   set.seed(1)
#'   x = matrix(runif(100000), nrow = 100, ncol = 1000)
#'
#'   y = runif(100)
#'
#'   fast_RGF_regr = FastRGF_Regressor$new(max_leaf = 50)
#'
#'   fast_RGF_regr$fit(x, y)
#'
#'   preds = fast_RGF_regr$predict(x)
#' }

FastRGF_Regressor <- R6::R6Class(
    "FastRGF_Regressor",
    inherit = Internal_class,
    lock_objects = FALSE,
    public = list(

        initialize = function(n_estimators = 500
                              , max_depth = 6
                              , max_leaf = 50
                              , tree_gain_ratio = 1.0
                              , min_samples_leaf = 5
                              , l1 = 1.0
                              , l2 = 1000.0
                              , opt_algorithm = "rgf"
                              , learning_rate = 0.001
                              , max_bin = NULL
                              , min_child_weight = 5.0
                              , data_l2 = 2.0
                              , sparse_max_features = 80000
                              , sparse_min_occurences = 5
                              , n_jobs = 1
                              , verbose = 0
        ) {

            # exceptions for 'min_samples_leaf', 'max_bin'
            #---------------------------------------------

            # must be either > as.integer(1.0) or in (0, 0.5]
            if (min_samples_leaf >= 1.0) {
                min_samples_leaf <- as.integer(min_samples_leaf)
            }

            # must be either NULL or an integer
            if (!is.null(max_bin)) {
                max_bin <- as.integer(max_bin)
            }

            # initialize FastRGF_Regressor
            #-----------------------------
            private$rgf_init <- RGF_mod$FastRGFRegressor(
                as.integer(n_estimators)
                , as.integer(max_depth)
                , as.integer(max_leaf)
                , tree_gain_ratio
                , min_samples_leaf
                , l1
                , l2
                , opt_algorithm
                , learning_rate
                , max_bin
                , min_child_weight
                , data_l2
                , as.integer(sparse_max_features)
                , as.integer(sparse_min_occurences)
                , as.integer(n_jobs)
                , as.integer(verbose)
            )
        }
    )
)


#' A Fast Regularized Greedy Forest classifier
#'
#'
#' @param x an R matrix (object) or a Python sparse matrix (object) of shape c(n_samples, n_features). The training input samples. The sparse matrix should be a Python sparse matrix. The helper functions \emph{mat_2scipy_sparse} and \emph{TO_scipy_sparse} allow the user to convert an R dense or sparse matrix to a scipy sparse matrix.
#' @param y a vector of shape c(n_samples). The target values (real numbers in regression).
#' @param n_estimators an integer. The number of trees in the forest (Original name: forest.ntrees.)
#' @param max_depth an integer. Maximum tree depth (Original name: dtree.max_level.)
#' @param max_leaf an integer. Maximum number of leaf nodes in best-first search (Original name: dtree.max_nodes.)
#' @param tree_gain_ratio a float. New tree is created when leaf-nodes gain < this value * estimated gain of creating new tree (Original name: dtree.new_tree_gain_ratio.)
#' @param min_samples_leaf an integer or float. Minimum number of training data points in each leaf node. If an integer, then consider min_samples_leaf as the minimum number. If a float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node (Original name: dtree.min_sample.)
#' @param loss a character string. One of \emph{"LS"} (Least squares loss), \emph{"MODLS"} (Modified least squares loss) or \emph{"LOGISTIC"} (Logistic loss) (Original name: dtree.loss.)
#' @param l1 a float. Used to control the degree of L1 regularization (Original name: dtree.lamL1.)
#' @param l2 a float. Used to control the degree of L2 regularization (Original name: dtree.lamL2.)
#' @param opt_algorithm a character string. Either \emph{"rgf"} or \emph{"epsilon-greedy"}. Optimization method for training forest (Original name: forest.opt.)
#' @param learning_rate a float. Step size of epsilon-greedy boosting. Meant for being used with opt_algorithm = "epsilon-greedy" (Original name: forest.stepsize.)
#' @param max_bin an integer or NULL. Maximum number of discretized values (bins). If NULL, 65000 is used for dense data and 200 for sparse data (Original name: discretize.(sparse/dense).max_buckets.)
#' @param min_child_weight a float. Minimum sum of data weights for each discretized value (bin) (Original name: discretize.(sparse/dense).min_bucket_weights.)
#' @param data_l2 a float. Used to control the degree of L2 regularization for discretization (Original name: discretize.(sparse/dense).lamL2.)
#' @param sparse_max_features an integer. Maximum number of selected features. Meant for being used with sparse data (Original name: discretize.sparse.max_features.)
#' @param sparse_min_occurences an integer. Minimum number of occurrences for a feature to be selected. Meant for being used with sparse data (Original name: discretize.sparse.min_occrrences.)
#' @param calc_prob a character string. Either \emph{"sigmoid"} or \emph{"softmax"}. Method of probability calculation
#' @param n_jobs an integer. The number of jobs to run in parallel for both fit and predict. If -1, all CPUs are used. If -2, all CPUs but one are used. If < -1, (n_cpus + 1 + n_jobs) are used (Original name: set.nthreads.)
#' @param verbose an integer. Controls the verbosity of the tree building process (Original name: set.verbose.)
#' @export
#' @details
#'
#' the \emph{fit} function builds a classifier from the training set (x, y).
#'
#' the \emph{predict} function predicts the class for x.
#'
#' the \emph{predict_proba} function predicts class probabilities for x.
#'
#' the \emph{cleanup} function removes tempfiles used by this model. See the issue \emph{https://github.com/RGF-team/rgf/issues/75}, which explains in which cases the \emph{cleanup} function applies.
#'
#' the \emph{get_params} function returns the parameters of the model.
#'
#' the \emph{score} function returns the mean accuracy on the given test data and labels.
#'
#' @references \emph{https://github.com/RGF-team/rgf/tree/master/python-package}, \emph{Tong Zhang, FastRGF: Multi-core Implementation of Regularized Greedy Forest (https://github.com/RGF-team/rgf/tree/master/FastRGF)}
#' @docType class
#' @importFrom R6 R6Class
#' @import reticulate
#' @section Methods:
#'
#' \describe{
#'  \item{\code{FastRGF_Classifier$new(n_estimators = 500, max_depth = 6,
#'                                     max_leaf = 50, tree_gain_ratio = 1.0,
#'                                     min_samples_leaf = 5, loss = "LS", l1 = 1.0,
#'                                     l2 = 1000.0, opt_algorithm = "rgf",
#'                                     learning_rate = 0.001, max_bin = NULL,
#'                                     min_child_weight = 5.0, data_l2 = 2.0,
#'                                     sparse_max_features = 80000,
#'                                     sparse_min_occurences = 5,
#'                                     calc_prob="sigmoid", n_jobs = 1,
#'                                     verbose = 0)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{fit(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{predict(x)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{predict_proba(x)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{cleanup()}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{get_params(deep = TRUE)}}{}
#'
#'  \item{\code{--------------}}{}
#'
#'  \item{\code{score(x, y, sample_weight = NULL)}}{}
#'
#'  \item{\code{--------------}}{}
#'  }
#'
#' @usage # init <- FastRGF_Classifier$new(n_estimators = 500, max_depth = 6,
#' #                                      max_leaf = 50, tree_gain_ratio = 1.0,
#' #                                      min_samples_leaf = 5, loss = "LS", l1 = 1.0,
#' #                                      l2 = 1000.0, opt_algorithm = "rgf",
#' #                                      learning_rate = 0.001, max_bin = NULL,
#' #                                      min_child_weight = 5.0, data_l2 = 2.0,
#' #                                      sparse_max_features = 80000,
#' #                                      sparse_min_occurences = 5,
#' #                                      calc_prob="sigmoid", n_jobs = 1,
#' #                                      verbose = 0)
#' @examples
#'
#' if (reticulate::py_available() && reticulate::py_module_available("rgf.sklearn")) {
#'
#'   library(RGF)
#'
#'   set.seed(1)
#'   x = matrix(runif(100000), nrow = 100, ncol = 1000)
#'
#'   y = sample(1:2, 100, replace = TRUE)
#'
#'   fast_RGF_class = FastRGF_Classifier$new(max_leaf = 50)
#'
#'   fast_RGF_class$fit(x, y)
#'
#'   preds = fast_RGF_class$predict_proba(x)
#' }

FastRGF_Classifier <- R6::R6Class(
    "FastRGF_Classifier",
    inherit = Internal_class,
    lock_objects = FALSE,
    public = list(

        initialize = function(n_estimators = 500
                              , max_depth = 6
                              , max_leaf = 50
                              , tree_gain_ratio = 1.0
                              , min_samples_leaf = 5
                              , loss = "LS"
                              , l1 = 1.0
                              , l2 = 1000.0
                              , opt_algorithm = "rgf"
                              , learning_rate = 0.001
                              , max_bin = NULL
                              , min_child_weight = 5.0
                              , data_l2 = 2.0
                              , sparse_max_features = 80000
                              , sparse_min_occurences = 5
                              , calc_prob = "sigmoid"
                              , n_jobs = 1
                              , verbose = 0
        ) {

            # exceptions for 'min_samples_leaf', 'max_bin'
            #---------------------------------------------

            # must be either > as.integer(1.0) or in (0, 0.5]
            if (min_samples_leaf >= 1.0) {
                min_samples_leaf <- as.integer(min_samples_leaf)
            }

            # must be either NULL or an integer
            if (!is.null(max_bin)) {
                max_bin <- as.integer(max_bin)
            }

            # initialize FastRGF_Classifier
            #------------------------------
            private$rgf_init = RGF_mod$FastRGFClassifier(
                as.integer(n_estimators)
                , as.integer(max_depth)
                , as.integer(max_leaf)
                , tree_gain_ratio
                , min_samples_leaf
                , loss
                , l1
                , l2
                , opt_algorithm
                , learning_rate
                , max_bin
                , min_child_weight
                , data_l2
                , as.integer(sparse_max_features)
                , as.integer(sparse_min_occurences)
                , calc_prob
                , as.integer(n_jobs)
                , as.integer(verbose)
            )
        }
    )
)
