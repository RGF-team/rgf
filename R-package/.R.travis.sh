mkdir -p $R_LIB_PATH
cd $TRAVIS_BUILD_DIR/R-package
echo "R_LIBS=$R_LIB_PATH" > .Renviron
echo 'options(repos = "https://cran.rstudio.com")' > .Rprofile

export PATH="$R_LIB_PATH/R/bin:$PATH"

sudo apt-get install gfortran-5
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-5 10
# use system-wide libraries (fix error "symbol _ZTINSt8ios_base7failureB5cxx11E, version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference")
conda remove --force libgfortran-ng libgcc-ng libstdcxx-ng

# install packages to build and check documentation
conda install --no-deps pandoc
sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-fonts-extra qpdf

# fix "libcurl error code 60: server certificate verification failed. CAfile: /etc/ssl/certs/ca-certificates.crt CRLfile: none"
echo -n | openssl s_client -connect arxiv.org:443 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' | sudo tee -a /etc/ssl/certs/ca-certificates.crt

if ! command -v R &> /dev/null; then
    R_VER=3.5.1
    cd $TRAVIS_BUILD_DIR
    wget https://cran.r-project.org/src/base/R-3/R-$R_VER.tar.gz
    tar -xzf R-$R_VER.tar.gz
    R-$R_VER/configure --enable-R-shlib --prefix=$R_LIB_PATH/R
    make
    make install
    cd $TRAVIS_BUILD_DIR/R-package
fi

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

echo "comment: off" > $TRAVIS_BUILD_DIR/codecov.yml
Rscript -e 'covr::codecov(quiet = FALSE)'
