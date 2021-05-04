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
    "xmlschema"
]

def installPackage(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    if os.environ["SKLEARN"] == "0.20.x":
        packages.insert(0,"scikit-learn==0.20.3")
    elif os.environ["SKLEARN"] == "0.23.x":
        packages.insert(0, "scikit-learn==0.23.1")
    for pck in packages:
        installPackage(pck)