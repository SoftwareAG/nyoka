from setuptools import setup, find_packages

exec(open("nyoka/metadata.py").read())

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
	long_description = f.read()


setup(
	name = "nyoka",
	version = __version__,
	description = 'A Python library to export Machine Learning/ Deep Learning models into PMML',
	long_description = long_description,
	long_description_content_type='text/markdown',
	author = "maintainer",
	author_email = "maintainer@nyoka.org",
	url = "https://github.com/nyoka-pmml/nyoka",
	license = __license__,
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
		"keras==2.2.4",
		"tensorflow==1.9.0",
		"statsmodels==0.9.0",
		"sklearn-pandas",
		"lightgbm"
	]
)
