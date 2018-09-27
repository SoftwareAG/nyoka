from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
	name = "nyoka",
	version = "1.1.0",
	description = 'A Python library to export Machine Learning/ Deep Learning models into PMML',
	long_description = long_description,
	long_description_content_type='text/markdown',
	author = "maintainer",
	author_email = "maintainer@nyoka.org",
	url = "https://github.com/nyoka-pmml/nyoka",
	license = "Apache Software License",
	classifiers = [
		"Development Status :: 5 - Production/Stable",
		"License :: OSI Approved :: Apache Software License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Intended Audience :: Developers",
		"Topic :: Scientific/Engineering :: Artificial Intelligence"
	],
	packages = find_packages(),
	install_requires = [
		"scikit-learn>=0.19.1",
		"keras==2.1.5",
		"tensorflow==1.9.0",
		"statsmodels==0.9.0",
		"sklearn-pandas",
		"lightgbm"
	]
)
