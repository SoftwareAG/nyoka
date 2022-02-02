import subprocess
import sys
import os

packages = [
    "pandas",
    "numpy",
    "statsmodels==0.11.1",
    "xgboost==1.5.2",
    "numpy==1.16.1",
    "glibc",
    "lxml",
    "sklearn-pandas",
    "lightgbm",
    "pytest-cov",
    "pytest",
    "codecov",
    "xmlschema",
    "scikit-learn==0.23.1"
]

def installPackage(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    for pck in packages:
        installPackage(pck)
