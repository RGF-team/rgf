sudo apt-get -qq install r-base-dev r-recommended pandoc

mkdir -p $R_LIB_PATH
cd $TRAVIS_BUILD_DIR/R-package
echo "R_LIBS=$R_LIB_PATH" > .Renviron
echo 'options(repos = "http://cran.rstudio.com")' > .Rprofile

Rscript -e 'if(!"devtools" %in% rownames(installed.packages())) { install.packages("devtools", dependencies = TRUE) }'
Rscript -e 'if(!"roxygen2" %in% rownames(installed.packages())) { install.packages("roxygen2", dependencies = TRUE) }'
Rscript -e 'if(!"testthat" %in% rownames(installed.packages())) { install.packages("testthat", dependencies = TRUE) }'
Rscript -e 'if(!"knitr" %in% rownames(installed.packages())) { install.packages("knitr", dependencies = TRUE) }'
Rscript -e 'if(!"covr" %in% rownames(installed.packages())) { install.packages("covr", dependencies = TRUE) }'
Rscript -e 'if(!"rmarkdown" %in% rownames(installed.packages())) { install.packages("rmarkdown", dependencies = TRUE) }'
Rscript -e 'if(!"reticulate" %in% rownames(installed.packages())) { install.packages("reticulate", dependencies = TRUE) }'
Rscript -e 'if(!"R6" %in% rownames(installed.packages())) { install.packages("R6", dependencies = TRUE) }'
Rscript -e 'if(!"Matrix" %in% rownames(installed.packages())) { install.packages("Matrix", dependencies = TRUE) }'

Rscript -e 'update.packages(ask = FALSE, instlib = Sys.getenv("R_LIB_PATH"))'

Rscript -e 'devtools::install_deps(pkg = ".", dependencies = TRUE)'

R CMD build . --no-build-vignettes --no-manual  || exit -1

PKG_FILE_NAME=$(ls -1t *.tar.gz | head -n 1)
R CMD check "${PKG_FILE_NAME}" --no-build-vignettes --no-manual --as-cran || exit -1

Rscript -e 'covr::codecov(quiet = FALSE)'
