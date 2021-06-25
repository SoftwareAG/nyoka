import subprocess
import sys
import os

packages = [
    "statsmodels==0.11.1",
    "xgboost==0.90",
    "numpy==1.16.1",
    "glibc",
    "lxml",
    "sklearn-pandas",
    "lightgbm",
    "pandas",
    "numpy",
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