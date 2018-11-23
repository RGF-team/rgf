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
                n_estimators = as.integer(n_estimators)
                , max_depth = as.integer(max_depth)
                , max_leaf = as.integer(max_leaf)
                , tree_grain_ratio = tree_gain_ratio
                , min_samples_leaf = min_samples_leaf
                , l1 = l1
                , l2 = l2
                , opt_algorithm = opt_algorithm
                , learning_rate = learning_rate
                , max_bin = max_bin
                , min_child_weight = min_child_weight
                , data_l2 = data_l2
                , sparse_max_features = as.integer(sparse_max_features)
                , sparse_min_features = as.integer(sparse_min_occurences)
                , n_jobs = as.integer(n_jobs)
                , verbose = as.integer(verbose)
            )
        }
    )
)
