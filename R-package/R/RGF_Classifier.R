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
#' @param init_model either NULL or a character string, optional (default=NULL). Filename of a previously saved model from which training should do warm-start. If model has been saved into multiple files, do not include numerical suffixes in the filename. \emph{NOTE:} Make sure you haven't forgotten to increase the value of the max_leaf parameter regarding to the specified warm-start model because warm-start model trees are counted in the overall number of trees.
#' @param filename a character string specifying a valid path to a file where the fitted model should be saved
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
#' the \emph{save_model} function saves a model to a file from which training can do warm-start in the future.
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
#'                                verbose = 0, init_model = NULL)}}{}
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
#'
#'  \item{\code{save_model(filename)}}{}
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
#' #                                  verbose = 0, init_model = NULL)
#' @examples
#'
#' try({
#'     if (reticulate::py_available(initialize = FALSE)) {
#'         if (reticulate::py_module_available("rgf.sklearn")) {
#'
#'             library(RGF)
#'
#'             set.seed(1)
#'             x = matrix(runif(1000), nrow = 100, ncol = 10)
#'
#'             y = sample(1:2, 100, replace = TRUE)
#'
#'             RGF_class = RGF_Classifier$new(max_leaf = 50)
#'
#'             RGF_class$fit(x, y)
#'
#'             preds = RGF_class$predict_proba(x)
#'         }
#'     }
#' }, silent = TRUE)

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
                              , init_model = NULL
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
                max_leaf = as.integer(max_leaf)
                , test_interval = as.integer(test_interval)
                , algorithm = algorithm
                , loss = loss
                , reg_depth = reg_depth
                , l2 = l2
                , sl2 = sl2
                , normalize = normalize
                , min_samples_leaf = min_samples_leaf
                , n_iter = n_iter
                , n_tree_search = as.integer(n_tree_search)
                , opt_interval = as.integer(opt_interval)
                , learning_rate = learning_rate
                , calc_prob = calc_prob
                , n_jobs = as.integer(n_jobs)
                , memory_policy = memory_policy
                , verbose = as.integer(verbose)
                , init_model = init_model
            )
        }
    )
)
