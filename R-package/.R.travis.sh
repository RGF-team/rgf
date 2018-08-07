mkdir -p $R_LIB_PATH
cd $TRAVIS_BUILD_DIR/R-package
echo "R_LIBS=$R_LIB_PATH" > .Renviron
echo 'options(repos = "http://cran.rstudio.com")' > .Rprofile

sudo apt-get install texlive-latex-recommended texlive-fonts-recommended qpdf  # packages to build and check documentation
conda install -c r --no-deps r-base _r-mutex pcre icu libcurl pandoc  # set up minimal R environment

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

R CMD build . || exit -1

PKG_FILE_NAME=$(ls -1t *.tar.gz | head -n 1)
PKG_NAME="${PKG_FILE_NAME%%_*}"
LOG_FILE_NAME="$PKG_NAME.Rcheck/00check.log"

R CMD check "${PKG_FILE_NAME}" --as-cran || exit -1
if grep -q -R "WARNING" "$LOG_FILE_NAME"; then
    echo "WARNINGS have been found in the build log!"
    exit -1
elif grep -q -R "NOTE" "$LOG_FILE_NAME"; then
    echo "NOTES have been found in the build log!"
    exit -1
fi

Rscript -e 'covr::codecov(quiet = FALSE)'
