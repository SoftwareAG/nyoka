import subprocess
import sys
import os

packages = [
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "xgboost==1.5.2",
    "glibc",
    "lxml",
    "sklearn-pandas",
    "lightgbm",
    "pytest-cov",
    "pytest",
    "codecov",
    "xmlschema",
    "scikit-learn"
]


def installPackage(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    for pck in packages:
        installPackage(pck)
