context('rgf R-package tests')

#========================================================================================  
# helper function to skip tests if we don't have the 'foo' module  [  https://github.com/rstudio/reticulate ]

skip_test_if_no_module <- function(MODULE) {                                              # MODULE is of type character string ( length(MODULE) >= 1 )

  if (length(MODULE) == 1) {

    module_exists <- reticulate::py_module_available(MODULE)}

  else {

    module_exists <- sum(as.vector(sapply(MODULE, function(x) reticulate::py_module_available(x)))) == length(MODULE)
  }

  if (!module_exists) {

    testthat::skip(paste0(MODULE, " is not available for testthat-testing"))
  }
}

#===========================================================================================    
# Input data

# data [ regression and (multiclass-) classification RGF_Regressor, RGF_Classifier ]
#-----------------------------------------------------------------------------------

set.seed(1)
x_rgf = matrix(runif(1000), nrow = 100, ncol = 10)


# data [ regression and (multiclass-) classification FastRGF_Regressor, FastRGF_Classifier ]
#-------------------------------------------------------------------------------------------

set.seed(2)
x_FASTrgf = matrix(runif(100000), nrow = 100, ncol = 1000)           # high dimensionality for 'FastRGF'    (however more observations are needed so that it works properly)


# response regression
#--------------------

set.seed(3)
y_reg = runif(100)


# response "binary" classification
#---------------------------------

set.seed(4)
y_BINclass = sample(1:2, 100, replace = T)


# response "multiclass" classification
#-------------------------------------

set.seed(5)
y_MULTIclass = sample(1:5, 100, replace = T)


# weights for the fit function
#------------------------------

set.seed(6)
W = runif(100)

#===========================================================================================  
# Tests for 'RGF_Regressor' & 'RGF_Classifier'

# tests for 'RGF_Regressor'
#-------------------------

testthat::test_that("the methods of the 'RGF_Regressor' class return the correct output", {

  skip_test_if_no_module("rgf.sklearn")

  init_regr = RGF_Regressor$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)

  init_regr$fit(x = x_rgf, y = y_reg, sample_weight = W)                    # include also a vector of weights

  pr = init_regr$predict(x_rgf)

  params = unlist(init_regr$get_params(deep = TRUE))

  validate = names(params) %in% c("normalize", "loss", "verbose", "algorithm", "n_iter", "learning_rate",
                             "sl2", "min_samples_leaf", "opt_interval", "l2", "n_tree_search",
                             "reg_depth", "memory_policy", "test_interval", "max_leaf")

  tmp_score = init_regr$score(x = x_rgf, y = y_reg)

  tmp_score_W = init_regr$score(x = x_rgf, y = y_reg, sample_weight = W)

  testthat::expect_true( length(pr) == length(y_reg) && sum(validate) == 15 && is.double(tmp_score) && is.double(tmp_score_W) )
})



# tests for 'RGF_Classifier'
#---------------------------

testthat::test_that("the methods of the 'RGF_Classifier' class return the correct output (binary classification)", {

  skip_test_if_no_module("rgf.sklearn")

  init_class = RGF_Classifier$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)

  init_class$fit(x = x_rgf, y = y_BINclass, sample_weight = W)                    # include also a vector of weights

  pr = init_class$predict(x_rgf)

  pr_proba = init_class$predict_proba(x_rgf)

  params = unlist(init_class$get_params(deep = TRUE))

  validate = names(params) %in% c("normalize", "loss", "verbose", "algorithm", "n_iter", "learning_rate",
                                  "sl2", "min_samples_leaf", "opt_interval", "l2", "n_tree_search",
                                  "reg_depth", "memory_policy", "test_interval", "max_leaf")

  tmp_score = init_class$score(x = x_rgf, y = y_BINclass)

  tmp_score_W = init_class$score(x = x_rgf, y = y_BINclass, sample_weight = W)

  testthat::expect_true( length(pr) == length(y_BINclass) && sum(validate) == 15 && is.double(tmp_score) && is.double(tmp_score_W) && ncol(pr_proba) == length(unique(y_BINclass)) )
})



testthat::test_that("the methods of the 'RGF_Classifier' class return the correct output (multiclass classification)", {

  skip_test_if_no_module("rgf.sklearn")

  init_class = RGF_Classifier$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)

  init_class$fit(x = x_rgf, y = y_MULTIclass, sample_weight = W)                    # include also a vector of weights

  pr = init_class$predict(x_rgf)

  pr_proba = init_class$predict_proba(x_rgf)

  params = unlist(init_class$get_params(deep = TRUE))

  validate = names(params) %in% c("normalize", "loss", "verbose", "algorithm", "n_iter", "learning_rate",
                                  "sl2", "min_samples_leaf", "opt_interval", "l2", "n_tree_search",
                                  "reg_depth", "memory_policy", "test_interval", "max_leaf")

  tmp_score = init_class$score(x = x_rgf, y = y_MULTIclass)

  tmp_score_W = init_class$score(x = x_rgf, y = y_MULTIclass, sample_weight = W)

  testthat::expect_true( length(pr) == length(y_BINclass) && sum(validate) == 15 && is.double(tmp_score) && is.double(tmp_score_W) && ncol(pr_proba) == length(unique(y_MULTIclass)) )
})


#===========================================================================================  
# Tests for 'FastRGF_Regressor' & 'FastRGF_Classifier'

# tests for 'FastRGF_Regressor'
#------------------------------

testthat::test_that("the methods of the 'FastRGF_Regressor' class return the correct output", {

  skip_test_if_no_module("rgf.sklearn")

  init_regr = FastRGF_Regressor$new(n_estimators = 50, max_bin = 65000)

  init_regr$fit(x = x_FASTrgf, y = y_reg, sample_weight = W)                    # include also a vector of weights

  pr = init_regr$predict(x_FASTrgf)

  params = unlist(init_regr$get_params(deep = TRUE))

  validate = names(params) %in% c("sparse_min_occurences", "n_jobs", "verbose", "learning_rate",
                                  "max_bin", "data_l2", "min_samples_leaf", "n_estimators", "sparse_max_features",
                                  "max_leaf", "opt_algorithm", "tree_gain_ratio", "min_child_weight",
                                  "l2", "l1", "max_depth")

  # here the FastRGF_Regressor returns a negative score mainly because the algorithm is meant to work best with many instances (here I
  # use only 100 for testing, and If I would increase the number of observations to 1000 then it would work as expected)

  tmp_score = init_regr$score(x = x_FASTrgf, y = y_reg)

  tmp_score_W = init_regr$score(x = x_FASTrgf, y = y_reg, sample_weight = W)

  testthat::expect_true( length(pr) == length(y_reg) && sum(validate) == 16 && is.double(tmp_score) && is.double(tmp_score_W) )
})




# tests for 'FastRGF_Classifier'
#------------------------------

testthat::test_that("the methods of the 'FastRGF_Classifier' class return the correct output (binary classification)", {

  skip_test_if_no_module("rgf.sklearn")

  init_class = FastRGF_Classifier$new(n_estimators = 50, max_bin = 65000)

  init_class$fit(x = x_FASTrgf, y = y_BINclass, sample_weight = W)                    # include also a vector of weights

  pr = init_class$predict(x_FASTrgf)

  pr_proba = init_class$predict_proba(x_FASTrgf)

  params = unlist(init_class$get_params(deep = TRUE))

  validate = names(params) %in% c("sparse_min_occurences", "n_jobs", "verbose", "learning_rate",
                                  "max_bin", "data_l2", "min_samples_leaf", "n_estimators", "sparse_max_features",
                                  "max_leaf", "opt_algorithm", "tree_gain_ratio", "min_child_weight",
                                  "l2", "l1", "max_depth")

  tmp_score = init_class$score(x = x_FASTrgf, y = y_BINclass)

  tmp_score_W = init_class$score(x = x_FASTrgf, y = y_BINclass, sample_weight = W)

  testthat::expect_true( length(pr) == length(y_BINclass) && sum(validate) == 16 && is.double(tmp_score) && is.double(tmp_score_W) && ncol(pr_proba) == length(unique(y_BINclass)) )
})



testthat::test_that("the methods of the 'FastRGF_Classifier' class return the correct output (multiclass classification)", {

  skip_test_if_no_module("rgf.sklearn")

  init_class = FastRGF_Classifier$new(n_estimators = 50, max_bin = 65000)

  init_class$fit(x = x_FASTrgf, y = y_MULTIclass, sample_weight = W)                    # include also a vector of weights

  pr = init_class$predict(x_FASTrgf)

  pr_proba = init_class$predict_proba(x_FASTrgf)

  params = unlist(init_class$get_params(deep = TRUE))

  validate = names(params) %in% c("sparse_min_occurences", "n_jobs", "verbose", "learning_rate",
                                  "max_bin", "data_l2", "min_samples_leaf", "n_estimators", "sparse_max_features",
                                  "max_leaf", "opt_algorithm", "tree_gain_ratio", "min_child_weight",
                                  "l2", "l1", "max_depth")

  tmp_score = init_class$score(x = x_FASTrgf, y = y_MULTIclass)

  tmp_score_W = init_class$score(x = x_FASTrgf, y = y_MULTIclass, sample_weight = W)

  testthat::expect_true( length(pr) == length(y_MULTIclass) && sum(validate) == 16 && is.double(tmp_score) && is.double(tmp_score_W) && ncol(pr_proba) == length(unique(y_MULTIclass)) )
})


#=========================================================================================== 
# Tests for scipy sparse


# conversion of an R matrix to a scipy sparse matrix
#---------------------------------------------------

testthat::test_that("the 'mat_2scipy_sparse' returns an error in case that the data is not inheriting matrix class", {

  skip_test_if_no_module("scipy")

  x_rgf_invalid = as.data.frame(x_rgf)

  testthat::expect_error( mat_2scipy_sparse(x_rgf_invalid) )
})


testthat::test_that("the 'mat_2scipy_sparse' returns an error in case that the 'format' parameter is invalid", {

  skip_test_if_no_module("scipy")

  testthat::expect_error( mat_2scipy_sparse(x_rgf, format = 'invalid') )
})


testthat::test_that("the 'mat_2scipy_sparse' returns a scipy CSR sparse matrix", {

  skip_test_if_no_module("scipy")

  res = mat_2scipy_sparse(x_rgf, format = 'sparse_row_matrix')

  same_dims = sum(unlist(reticulate::py_to_r(res$shape)) == dim(x_rgf)) == 2         # sparse matrix has same dimensions as input dense matrix

  testthat::expect_true( same_dims && inherits(res, "scipy.sparse.csr.csr_matrix")  )
})


testthat::test_that("the 'mat_2scipy_sparse' returns a scipy CSC sparse matrix", {

  skip_test_if_no_module("scipy")

  res = mat_2scipy_sparse(x_rgf, format = 'sparse_column_matrix')

  same_dims = sum(unlist(reticulate::py_to_r(res$shape)) == dim(x_rgf)) == 2         # sparse matrix has same dimensions as input dense matrix

  testthat::expect_true( same_dims && inherits(res, "scipy.sparse.csc.csc_matrix") )
})



# run the following tests on all operating systems except for 'Macintosh'
# [ otherwise it will raise an error due to the fact that the 'scipy-sparse' library ( applied on 'TO_scipy_sparse' function)
#   on CRAN is not upgraded and the older version includes a bug ('TypeError : could not interpret data type') ]
# reference : https://github.com/scipy/scipy/issues/5353

if (Sys.info()["sysname"] != 'Darwin') {

  # conversion of an R 'dgCMatrix' and 'dgRMatrix' to a scipy sparse matrices
  #--------------------------------------------------------------------------

  testthat::test_that("the 'TO_scipy_sparse' returns an error in case that the input object is not of type 'dgCMatrix' or 'dgRMatrix'", {

    skip_test_if_no_module("scipy")

    mt = matrix(runif(20), nrow = 5, ncol = 4)

    testthat::expect_error( TO_scipy_sparse(mt) )
  })


  testthat::test_that("the 'TO_scipy_sparse' returns the correct output for dgCMatrix", {

    skip_test_if_no_module("scipy")

    data = c(1, 0, 2, 0, 0, 3, 4, 5, 6)

    dgcM = Matrix::Matrix(data = data, nrow = 3,

                          ncol = 3, byrow = TRUE,

                          sparse = TRUE)

    res = TO_scipy_sparse(dgcM)

    validate_dims = sum(dim(dgcM) == unlist(reticulate::py_to_r(res$shape))) == 2      # sparse matrix has same dimensions as input R sparse matrix

    testthat::expect_true( validate_dims && inherits(res, "scipy.sparse.csc.csc_matrix") )
  })


  testthat::test_that("the 'TO_scipy_sparse' returns the correct output for dgRMatrix", {

    skip_test_if_no_module("scipy")

    data = c(1, 0, 2, 0, 0, 3, 4, 5, 6)

    dgrM = as(Matrix::Matrix(data = data, nrow = 3,

                             ncol = 3, sparse = TRUE),

              "RsparseMatrix")

    res = TO_scipy_sparse(dgrM)

    validate_dims = sum(dim(dgrM) == unlist(reticulate::py_to_r(res$shape))) == 2      # sparse matrix has same dimensions as input R sparse matrix

    testthat::expect_true( validate_dims && inherits(res, "scipy.sparse.csr.csr_matrix") )
  })


  # test that one of the RGF classes works with sparse (scipy) matrices
  #--------------------------------------------------------------------

  testthat::test_that("the RGF_Regressor works with sparse (scipy) matrices", {

    skip_test_if_no_module(c("rgf.sklearn", 'scipy'))

    set.seed(1)
    sap = sapply(1:1000, function(x) sample(c(0.0, runif(1)), 1, replace = FALSE))            # create sparse data

    dgcM = Matrix::Matrix(data = sap,

                          nrow = 100, ncol = 10,

                          byrow = TRUE, sparse = TRUE)

    scipySprse = TO_scipy_sparse(dgcM)                                            # use scipy sparse matrix

    init_regr = RGF_Regressor$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)

    init_regr$fit(x = scipySprse, y = y_reg, sample_weight = W)                           # include also a vector of weights

    pr = init_regr$predict(scipySprse)

    params = unlist(init_regr$get_params(deep = TRUE))

    validate = names(params) %in% c("normalize", "loss", "verbose", "algorithm", "n_iter", "learning_rate",
                                    "sl2", "min_samples_leaf", "opt_interval", "l2", "n_tree_search",
                                    "reg_depth", "memory_policy", "test_interval", "max_leaf")

    tmp_score = init_regr$score(x = scipySprse, y = y_reg)

    tmp_score_W = init_regr$score(x = scipySprse, y = y_reg, sample_weight = W)

    testthat::expect_true( length(pr) == length(y_reg) && sum(validate) == 15 && is.double(tmp_score) && is.double(tmp_score_W) )
  })
}


#=========================================================================================== 
# test feature importances


testthat::test_that("the feature importances of the 'RGF_Regressor' class works as expected", {
  
  skip_test_if_no_module("rgf.sklearn")
  
  init_regr = RGF_Regressor$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)
  
  init_regr$fit(x = x_rgf, y = y_reg, sample_weight = W)                    # include also a vector of weights
  
  vec_imp = init_regr$feature_importances()

  testthat::expect_true( inherits(vec_imp, 'array') && length(vec_imp) == ncol(x_rgf) )
})


#===========================================================================================
# test dump-model


testthat::test_that("the 'dump_model' method returns the correct output (Dumps the forest information to the R session -- works ONLY for RGF and NOT for FastRGF)", {
  
  skip_test_if_no_module("rgf.sklearn")
  
  init_class = RGF_Classifier$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)
  
  init_class$fit(x = x_rgf, y = y_BINclass)
  
  dump_model = init_class$dump_model()
  
  #---------------------------------------
  # for pretty-print in the R session use:  print( dump_model() )
  #---------------------------------------
  
  output_dump = reticulate::py_capture_output(dump_model())
  
  #-------------------------------------------------
  # function to search for terms in the dumped model
  #-------------------------------------------------
  
  search_for_term = function(term, model_dump) {
    
    regex = gregexpr(pattern = term, text = model_dump)[[1]]
    
    len_result = attributes(regex)$match.length
    regex = as.vector(regex)
    
    term_results = unlist(lapply(1:length(regex), function(x) {
      
      substr(model_dump, start = regex[x], stop = regex[x] + len_result[x] - 1)
    }))
    
    return(all(term_results == term))
  }
  
  is_depth_0_in_model_dump = search_for_term(term = 'depth=0', model_dump = output_dump)
  is_gain_0_in_model_dump = search_for_term(term = 'gain=0', model_dump = output_dump)
  
  is_depth_1_in_model_dump = search_for_term(term = 'depth=1', model_dump = output_dump)
  is_gain_1_in_model_dump = search_for_term(term = 'gain=1', model_dump = output_dump)
  
  testthat::expect_true( nchar(output_dump) > 0 && object.size(output_dump) > 0 && is_depth_0_in_model_dump && 
                           is_gain_0_in_model_dump && is_depth_1_in_model_dump && is_gain_1_in_model_dump )
})


#===========================================================================================
# test saving a model


testthat::test_that("the 'save_model' method returns the correct output -- works ONLY for RGF and NOT for FastRGF", {
  
  skip_test_if_no_module("rgf.sklearn")
  
  init_class = RGF_Classifier$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)
  
  init_class$fit(x = x_rgf, y = y_BINclass)
      
  tmp_file = tempfile(fileext = '.model')
  
  SIZE_begin = file.info(tmp_file)$size
  
  sv_md = init_class$save_model(filename = tmp_file)
  
  SIZE_after = file.info(tmp_file)$size                                   # size of the saved model
  
  binary_file = file(tmp_file, "rb")                                      # connection to binary file
  raw_binary = readBin(binary_file, character(), n = 10000)               # read first 10.000 characters of the binary file
  close(binary_file)                                                      # close connection
  
  idx_chars = which(raw_binary != "")                                     # keep non-empty strings
  raw_binary = raw_binary[idx_chars]
  
  idx_max_leaf = which(gregexpr('max_leaf_forest', raw_binary) != -1)     # search for 'max_leaf_forest'
  idx_sl2 = which(gregexpr('reg_sL2', raw_binary) != -1)                  # search for 'reg_sL2'
  
  if (file.exists(tmp_file)) file.remove(tmp_file)
  
    testthat::expect_true( is.na(SIZE_begin) && (SIZE_after > 0) && (length(idx_max_leaf) > 0) && (length(idx_sl2) > 0) )
})


#=========================================================================================== 
# test the 'cleanup' method


  
testthat::test_that("the 'cleanup' method (ESTIMATOR specific) works as expected for both RGF and FastRGF (checking of the size of the temporary directory before and after the '$fit' method)", {
  
  skip_test_if_no_module("rgf.sklearn")
  
  #-------------------------------------------------------------------------------- 
  # default directory where the temporary 'rgf' files are saved 
  
  default_dir = file.path(dirname(tempdir()), 'rgf')
  
  #-------------------------------------------------------------------------------- 
  # RGF
  
  lst_files_tmp = list.files(path = default_dir)

  init_class = RGF_Classifier$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)
  init_class$fit(x = x_rgf, y = y_BINclass)
  
  lst_files_tmp_upd_rgf = list.files(path = default_dir)
  
  init_exists_upd_rgf = (dir.exists(default_dir) == TRUE)
  init_num_files_rgf = length(lst_files_tmp_upd_rgf)
  
  init_class$cleanup()
  
  lst_files_tmp_upd_rgf = list.files(path = default_dir)
  init_num_files_rgf_after_clean = length(lst_files_tmp_upd_rgf)
  
  end_state_rgf = (init_num_files_rgf_after_clean < init_num_files_rgf)
  
  #-------------------------------------------------------------------------------- 
  # FastRGF
  
  init_class = FastRGF_Classifier$new(n_estimators = 50, max_bin = 65000)
  init_class$fit(x = x_FASTrgf, y = y_MULTIclass)
  
  lst_files_tmp_upd_fastrgf = list.files(path = default_dir)
  init_num_files_fastrgf = length(lst_files_tmp_upd_fastrgf)
  
  init_class$cleanup()
  
  lst_files_tmp_upd_fastrgf = list.files(path = default_dir)
  init_num_files_fastrgf_after_clean = length(lst_files_tmp_upd_fastrgf)
  
  end_state_fastrgf = (init_num_files_fastrgf_after_clean < init_num_files_fastrgf)
  
  testthat::expect_true( init_exists_upd_rgf && end_state_rgf && end_state_fastrgf )
})



testthat::test_that("the 'cleanup' method (APPLIES TO ALL ESTIMATORS) works as expected for both RGF and FastRGF (checking of the size of the temporary directory before and after the '$fit' method)", {
  
  skip_test_if_no_module("rgf.sklearn")
  
  #-------------------------------------------------------------------------------- 
  # default directory where the temporary 'rgf' files are saved 
  
  default_dir = file.path(dirname(tempdir()), 'rgf')
  
  #-------------------------------------------------------------------------------- 
  # RGF
  
  lst_files_tmp = list.files(path = default_dir)

  init_class = RGF_Classifier$new(max_leaf = 50, sl2 = 0.1, n_iter = 10)
  init_class$fit(x = x_rgf, y = y_BINclass)
  
  lst_files_tmp_upd_rgf = list.files(path = default_dir)
  
  init_exists_upd_rgf = (dir.exists(default_dir) == TRUE)
  init_num_files_rgf = length(lst_files_tmp_upd_rgf)

  #-------------------------------------------------------------------------------- 
  # FastRGF
  
  init_class = FastRGF_Classifier$new(n_estimators = 50, max_bin = 65000)
  init_class$fit(x = x_FASTrgf, y = y_MULTIclass)
  
  lst_files_tmp_upd_fastrgf = list.files(path = default_dir)
  init_num_files_fastrgf = length(lst_files_tmp_upd_fastrgf)
  
  RGF_cleanup_temp_files()
  
  lst_files_tmp_end_state = list.files(path = default_dir)
  
  testthat::expect_true( init_exists_upd_rgf && (init_num_files_rgf > 0) && (init_num_files_fastrgf > init_num_files_rgf) &&
                            ( init_num_files_rgf > length(lst_files_tmp) && init_num_files_fastrgf > length(lst_files_tmp_end_state) ) )       # normally, both initial and end state must have the same length [ length(lst_files_tmp) == length(lst_files_tmp_end_state) ]
})

