#!/bin/bash

source activate $CONDA_ENV

cd $GITHUB_WORKSPACE/R-package
if [[ $OS_NAME == "macos-latest" ]]; then
  brew install r qpdf pandoc
  brew install --cask basictex
  export PATH="/Library/TeX/texbin:$PATH"
  sudo tlmgr --verify-repo=none update --self
  sudo tlmgr --verify-repo=none install inconsolata helvetic

  echo 'options(pkgType = "mac.binary")' > .Rprofile
  echo 'options(install.packages.check.source = "no")' >> .Rprofile
else
  tlmgr --verify-repo=none update --self
  tlmgr --verify-repo=none install ec
fi

R_LIB_PATH=$HOME/R
mkdir -p $R_LIB_PATH
echo "R_LIBS=$R_LIB_PATH" > .Renviron

# ignore R CMD CHECK NOTE checking how long it has
# been since the last submission
export _R_CHECK_CRAN_INCOMING_REMOTE_=0

if [[ $OS_NAME == "macos-latest" ]]; then
  Rscript -e "install.packages('devtools', dependencies = TRUE, repos = 'https://cran.r-project.org')"
fi
Rscript -e 'devtools::install_deps(pkg = ".", dependencies = TRUE)'

R CMD build . || exit -1

PKG_FILE_NAME=$(ls -1t *.tar.gz | head -n 1)
PKG_NAME="${PKG_FILE_NAME%%_*}"
LOG_FILE_NAME="$PKG_NAME.Rcheck/00check.log"
COVERAGE_FILE_NAME="$PKG_NAME.Rcheck/coverage.log"

R CMD check "${PKG_FILE_NAME}" --as-cran || exit -1
if grep -q -E "NOTE|WARNING|ERROR" "$LOG_FILE_NAME"; then
    echo "NOTEs, WARNINGs or ERRORs have been found by R CMD check"
    exit -1
fi

Rscript -e 'covr::codecov(quiet = FALSE)' 2>&1 | tee "$COVERAGE_FILE_NAME"
if [[ "$(grep -R "RGF Coverage:" $COVERAGE_FILE_NAME | rev | cut -d" " -f1 | rev | cut -d"." -f1)" -le 50 ]]; then
  echo "Code coverage is extremely small!"
  exit -1
fi
