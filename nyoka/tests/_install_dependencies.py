import subprocess
import sys
import os

packages = [
    "scikit-learn==0.20.3",
    "keras==2.2.4",
    "tensorflow==1.9.0",
    "statsmodels==0.9.0",
    "xgboost==0.82",
    "glibc",
    "lxml",
    "sklearn-pandas",
    "lightgbm",
    "h5py",
    "pandas",
    "numpy",
    "pytest-cov",
    "pytest",
    "codecov",
    "pillow",
    "keras-retinanet"
]
def installPackage(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    for pck in packages:
        installPackage(pck)