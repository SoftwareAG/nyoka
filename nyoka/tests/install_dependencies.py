import subprocess
import sys
import os

packages = [
    "scikit-learn>=0.19.1",
    "keras==2.2.4",
    "tensorflow==1.9.0",
    "statsmodels==0.9.0",
    "glibc",
    "lxml",
    "jupyter",
    "sklearn-pandas",
    "lightgbm",
    "h5py",
    "pandas",
    "numpy"
]
def installPackage(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    for pck in packages:
        print('\n',pck.capitalize(),'-->\n')
        installPackage(pck)