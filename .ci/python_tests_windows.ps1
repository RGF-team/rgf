function Check-Output {
    param( [bool]$Success )
    if (!$Success) {
        $host.SetShouldExit(-1)
        Exit -1
    }
}


$ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
$InstallerName = "$env:GITHUB_WORKSPACE\Miniconda3-latest-Windows-x86_64.exe"
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $InstallerName
Start-Process -FilePath $InstallerName -ArgumentList "/InstallationType=JustMe /RegisterPython=0 /S /D=$env:CONDA_PATH" -Wait
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda create -q -n $env:CONDA_ENV python=$env:PYTHON_VERSION joblib numpy scikit-learn scipy pandas pytest
conda activate $env:CONDA_ENV
cd $env:GITHUB_WORKSPACE\python-package
pytest tests -v ; Check-Output $?
python setup.py sdist --formats gztar ; Check-Output $?
pip install dist\rgf_python-$env:RGF_VER.tar.gz -v ; Check-Output $?
pytest tests -v ; Check-Output $?
