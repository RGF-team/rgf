
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
y_BINclass = sample(1:2, 100, replace = TRUE)


# response "multiclass" classification
#-------------------------------------

set.seed(5)
y_MULTIclass = sample(1:5, 100, replace = TRUE)


# weights for the fit function
#------------------------------

set.seed(6)
W = runif(100)
