mkdir -p $R_LIB_PATH
cd $TRAVIS_BUILD_DIR/R-package
echo "R_LIBS=$R_LIB_PATH" > .Renviron
echo 'options(repos = "https://cran.rstudio.com")' > .Rprofile
if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    echo 'options(pkgType = "mac.binary")' >> .Rprofile
fi
echo 'options(install.packages.check.source = "no")' >> .Rprofile

export PATH="$R_LIB_PATH/R/bin:$PATH"

if [[ $TRAVIS_OS_NAME == "linux" ]]; then
    sudo apt-get install gfortran-5 libcurl4-openssl-dev
    sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-5 10

    sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-fonts-extra qpdf

    if ! command -v R &> /dev/null; then
        R_VER=3.6.2
        cd $TRAVIS_BUILD_DIR
        wget https://cran.r-project.org/src/base/R-3/R-$R_VER.tar.gz
        tar -xzf R-$R_VER.tar.gz
        R-$R_VER/configure --enable-R-shlib --prefix=$R_LIB_PATH/R
        make
        make install
        cd $TRAVIS_BUILD_DIR/R-package
    fi
else
    brew install r qpdf
    brew cask install basictex
    export PATH="/Library/TeX/texbin:$PATH"
    sudo tlmgr update --self
    sudo tlmgr install inconsolata helvetic
fi

conda install -y --no-deps pandoc

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
COVERAGE_FILE_NAME="$PKG_NAME.Rcheck/coverage.log"

R CMD check "${PKG_FILE_NAME}" --as-cran || exit -1
if grep -q -R "WARNING" "$LOG_FILE_NAME"; then
    echo "WARNINGS have been found in the build log!"
    exit -1
fi

Rscript -e 'covr::codecov(quiet = FALSE)' 2>&1 | tee "$COVERAGE_FILE_NAME"
if [[ "$(grep -R "RGF Coverage:" $COVERAGE_FILE_NAME | rev | cut -d" " -f1 | rev | cut -d"." -f1)" -le 50 ]]; then
    echo "Code coverage is extremely small!"
    exit -1
fi
