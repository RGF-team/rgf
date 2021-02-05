function Check-Output {
    param( [bool]$Success )
    if (!$Success) {
        $host.SetShouldExit(-1)
        Exit -1
    }
}


Import-CliXml .\env-vars.clixml | % { Set-Item "env:$($_.Name)" $_.Value }
tzutil /s "GMT Standard Time"
cd $env:APPVEYOR_BUILD_FOLDER

[Void][System.IO.Directory]::CreateDirectory($env:R_LIB_PATH)

$env:PATH = "$env:R_LIB_PATH\Rtools\bin;" + "$env:R_LIB_PATH\R\bin\x64;" + "$env:R_LIB_PATH\miktex\texmfs\install\miktex\bin\x64;" + $env:PATH
$env:BINPREF = "C:/mingw-w64/x86_64-8.1.0-posix-seh-rt_v6-rev0/mingw64/bin/"

if (!(Get-Command R.exe -errorAction SilentlyContinue)) {

    $R_VER="3.6.2"
    appveyor DownloadFile https://cloud.r-project.org/bin/windows/base/old/$R_VER/R-$R_VER-win.exe -FileName ./R-win.exe
    Start-Process -FilePath .\R-win.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH\R /COMPONENTS=main,x64"

    appveyor DownloadFile https://cloud.r-project.org/bin/windows/Rtools/Rtools35.exe -FileName ./Rtools.exe
    Start-Process -FilePath .\Rtools.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH\Rtools"

    appveyor DownloadFile https://miktex.org/download/win/miktexsetup-x64.zip -FileName ./miktexsetup-x64.zip
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory("miktexsetup-x64.zip", "miktex")
    .\miktex\miktexsetup_standalone.exe --local-package-repository=.\miktex\download --package-set=essential --quiet download
    .\miktex\download\miktexsetup_standalone.exe --portable="$env:R_LIB_PATH\miktex" --quiet install
}

initexmf --set-config-value [MPM]AutoInstall=1
conda install -y --no-deps pandoc

cd .\R-package
Add-Content .Renviron "R_LIBS=$env:R_LIB_PATH"
Add-Content .Rprofile "options(repos = 'https://cran.rstudio.com')"
Add-Content .Rprofile "options(pkgType = 'binary')"
Add-Content .Rprofile "options(install.packages.check.source = 'no')"
Add-Content .Rprofile "Sys.setenv(RETICULATE_PYTHON = '$([RegEx]::Escape($env:CONDA_PREFIX))/python.exe')"

# Temporary remove coverage badge as it causes errors during checks
$README = Get-Content README.md
$README -Replace [regex]::Escape($README[1]) | Set-Content -Encoding UTF8 README.md

Rscript -e "if(!'devtools' %in% rownames(installed.packages())) { install.packages('devtools', dependencies = TRUE) }"
Rscript -e "if(!'roxygen2' %in% rownames(installed.packages())) { install.packages('roxygen2', dependencies = TRUE) }"
Rscript -e "if(!'testthat' %in% rownames(installed.packages())) { install.packages('testthat', dependencies = TRUE) }"
Rscript -e "if(!'knitr' %in% rownames(installed.packages())) { install.packages('knitr', dependencies = TRUE) }"
Rscript -e "if(!'covr' %in% rownames(installed.packages())) { install.packages('covr', dependencies = TRUE) }"
Rscript -e "if(!'rmarkdown' %in% rownames(installed.packages())) { install.packages('rmarkdown', dependencies = TRUE) }"
Rscript -e "if(!'reticulate' %in% rownames(installed.packages())) { install.packages('reticulate', dependencies = TRUE) }"
Rscript -e "if(!'R6' %in% rownames(installed.packages())) { install.packages('R6', dependencies = TRUE) }"
Rscript -e "if(!'Matrix' %in% rownames(installed.packages())) { install.packages('Matrix', dependencies = TRUE) }"

Rscript -e "update.packages(ask = FALSE, instlib = Sys.getenv('R_LIB_PATH'))"

Rscript -e "devtools::install_deps(pkg = '.', dependencies = TRUE)"

R.exe CMD build . ; Check-Output $?

$PKG_FILE_NAME = Get-Item *.tar.gz
$PKG_NAME = $PKG_FILE_NAME.BaseName.split("_")[0]
$LOG_FILE_NAME = "$PKG_NAME.Rcheck/00check.log"
$COVERAGE_FILE_NAME = "$PKG_NAME.Rcheck/coverage.log"

R.exe CMD check "${PKG_FILE_NAME}" --as-cran ; Check-Output $?
7z a R_logs.zip "$PKG_NAME.Rcheck/*"
appveyor PushArtifact R_logs.zip
if (Get-Content "$LOG_FILE_NAME" | Select-String -Pattern "WARNING" -Quiet) {
    echo "WARNINGS have been found in the build log!"
    Check-Output $False
}

Rscript -e "covr::codecov(quiet = FALSE)" *>&1 | Tee-Object "$COVERAGE_FILE_NAME"
$Coverage = 0
$Match = Get-Content "$COVERAGE_FILE_NAME" | Select-String -Pattern "RGF Coverage:" | Select-Object -First 1
$Coverage = [float]$Match.Line.Trim().Split(" ")[-1].Replace("%", "")
if ($Coverage -le 50) {
    echo "Code coverage is extremely small!"
    Check-Output $False
}
