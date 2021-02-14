function Check-Output {
    param( [bool]$Success )
    if (!$Success) {
        $host.SetShouldExit(-1)
        Exit -1
    }
}

$env:PATH += ";$env:CONDA_PATH;$env:CONDA_PATH\bin;$env:CONDA_PATH\condabin;$env:CONDA_PATH\Scripts"
$ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
$InstallerName = "$env:GITHUB_WORKSPACE\Miniconda3-latest-Windows-x86_64.exe"
echo $env:CONDA_PATH
mkdir $env:CONDA_PATH -Force
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $InstallerName
&$InstallerName /InstallationType=JustMe /RegisterPython=0 /D=$env:CONDA_PATH
cd $env:CONDA_PATH
dir
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda create -q -n $env:CONDA_ENV python=$env:PYTHON_VERSION joblib numpy scikit-learn scipy pandas pytest
activate $env:CONDA_ENV
cd $env:GITHUB_WORKSPACE\python-package
python setup.py sdist --formats gztar ; Check-Output $?
pip install dist\rgf_python-%RGF_VER%.tar.gz -v ; Check-Output $?
pytest tests -v ; Check-Output $?
