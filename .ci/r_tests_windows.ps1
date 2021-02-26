function Check-Output {
    param( [bool]$Success )
    if (!$Success) {
        $host.SetShouldExit(-1)
        Exit -1
    }
}


conda activate $env:CONDA_ENV

tzutil /s "GMT Standard Time"

$env:R_LIB_PATH = "$env:USERPROFILE\R"
$env:CTAN_PACKAGE_ARCHIVE = "https://ctan.math.illinois.edu/systems/win32/miktex/tm/packages/"
$env:PATH += ";$env:R_LIB_PATH\Rtools\usr\bin" + ";$env:R_LIB_PATH\Rtools\mingw64\bin" + ";$env:R_LIB_PATH\R\bin\x64" + ";$env:R_LIB_PATH\miktex\texmfs\install\miktex\bin\x64"
cd $env:GITHUB_WORKSPACE

[Void][System.IO.Directory]::CreateDirectory($env:R_LIB_PATH)
Remove-Item C:\rtools40 -Force -Recurse -ErrorAction Ignore

# ignore R CMD CHECK NOTE checking how long it has
# been since the last submission
$env:_R_CHECK_CRAN_INCOMING_ = 0
$env:_R_CHECK_CRAN_INCOMING_REMOTE_ = 0

$R_VER = "4.0.3"
$ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
Invoke-WebRequest -Uri https://cloud.r-project.org/bin/windows/base/old/$R_VER/R-$R_VER-win.exe -OutFile R-win.exe
Start-Process -FilePath R-win.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH\R /COMPONENTS=main,x64" ; Check-Output $?

Invoke-WebRequest -Uri https://cran.r-project.org/bin/windows/Rtools/rtools40-x86_64.exe -OutFile Rtools.exe
Start-Process -FilePath Rtools.exe -NoNewWindow -Wait -ArgumentList "/TYPE=full /VERYSILENT /SUPPRESSMSGBOXES /DIR=$env:R_LIB_PATH\Rtools" ; Check-Output $?

Invoke-WebRequest -Uri https://miktex.org/download/win/miktexsetup-x64.zip -OutFile miktexsetup-x64.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("miktexsetup-x64.zip", "miktex")
.\miktex\miktexsetup_standalone.exe --remote-package-repository="$env:CTAN_PACKAGE_ARCHIVE" --local-package-repository=.\miktex\download --package-set=essential --quiet download ; Check-Output $?
.\miktex\download\miktexsetup_standalone.exe --remote-package-repository="$env:CTAN_PACKAGE_ARCHIVE" --portable="$env:R_LIB_PATH\miktex" --quiet install ; Check-Output $?

initexmf --set-config-value [MPM]AutoInstall=1
echo yes | pacman -S mingw-w64-x86_64-qpdf

cd "$env:GITHUB_WORKSPACE\R-package"
Add-Content .Renviron "R_LIBS=$env:R_LIB_PATH"
Add-Content .Rprofile "options(repos = 'https://cran.r-project.org')"
Add-Content .Rprofile "options(pkgType = 'binary')"
Add-Content .Rprofile "options(install.packages.check.source = 'no')"

Rscript -e "install.packages('devtools', dependencies = TRUE)"
Rscript -e "devtools::install_deps(pkg = '.', dependencies = TRUE)"

R.exe CMD build . ; Check-Output $?

$PKG_FILE_NAME = Get-Item *.tar.gz
$PKG_NAME = $PKG_FILE_NAME.BaseName.split("_")[0]
$LOG_FILE_NAME = "$PKG_NAME.Rcheck/00check.log"
$COVERAGE_FILE_NAME = "$PKG_NAME.Rcheck/coverage.log"

R.exe CMD check "${PKG_FILE_NAME}" --as-cran ; Check-Output $?

if (Get-Content "$LOG_FILE_NAME" | Select-String -Pattern "NOTE|WARNING|ERROR" -CaseSensitive -Quiet) {
    echo "NOTEs, WARNINGs or ERRORs have been found by R CMD check"
    Check-Output $False
}

Rscript -e "covr::codecov(quiet = FALSE)" *>&1 | Tee-Object -FilePath "$COVERAGE_FILE_NAME"
#$Coverage = 0
#$Match = Get-Content "$COVERAGE_FILE_NAME" | Select-String -Pattern "RGF Coverage:" | Select-Object -First 1
#$Coverage = [float]$Match.Line.Trim().Split(" ")[-1].Replace("%", "")
#if ($Coverage -le 50) {
#    echo "Code coverage is extremely small!"
#    Check-Output $False
#}
