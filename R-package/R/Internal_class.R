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
        },
        
        # save_model
        #-----------
        save_model = function(filename) {
          private$rgf_init$save_model(filename)
          return(invisible(NULL))
        }
    ),

    private = list(
        rgf_init = NULL
    )
)
