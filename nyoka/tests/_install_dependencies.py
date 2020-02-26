import subprocess
import sys
import os

packages = [
    "scikit-learn==0.20.3",
    "statsmodels",
    "xgboost==0.82",
    "numpy==1.16.1",
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
    "keras-retinanet",
    "xmlschema"
]

packages_36 = [
    "keras==2.2.4",
    "tensorflow==1.9.0"
]

packages_37 = [
    "keras",
    "tensorflow==1.15.0"
]

def installPackage(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    print(sys.version)
    if sys.version_info[1] == 6:
        for pck in packages+packages_36:
            installPackage(pck)
    elif sys.version_info[1] == 7:
        for pck in packages+packages_37:
            installPackage(pck)