import subprocess
import sys
import os

packages = [
    "statsmodels==0.11.1",
    "xgboost==0.90",
    "keras==2.2.4",
    "numpy==1.16.1",
    "glibc",
    "lxml",
    "sklearn-pandas",
    "lightgbm",
    "h5py==2.10.0",
    "pandas",
    "numpy",
    "pytest-cov",
    "pytest",
    "codecov",
    "pillow",
    "keras-retinanet==0.5.1",
    "xmlschema"
]

packages_36 = [
    "tensorflow==1.9.0"
]

packages_37 = [
    "tensorflow==1.15.0"
]

def installPackage(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    if os.environ["SKLEARN"] == "0.20.x":
        packages.insert(0,"scikit-learn==0.20.3")
    elif os.environ["SKLEARN"] == "0.23.x":
        packages.insert(0, "scikit-learn==0.23.1")
    if sys.version_info[1] == 6:
        for pck in packages+packages_36:
            installPackage(pck)
    elif sys.version_info[1] == 7:
        for pck in packages+packages_37:
            installPackage(pck)