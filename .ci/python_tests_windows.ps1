function Check-Output {
    param( [bool]$Success )
    if (!$Success) {
        $host.SetShouldExit(-1)
        Exit -1
    }
}


conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda create -q -n $env:CONDA_ENV python=$env:PYTHON_VERSION joblib numpy scikit-learn scipy pandas pytest
activate $env:CONDA_ENV
cd $env:GITHUB_WORKSPACE\python-package
python setup.py sdist --formats gztar ; Check-Output $?
pip install dist\rgf_python-%RGF_VER%.tar.gz -v ; Check-Output $?
pytest tests -v ; Check-Output $?