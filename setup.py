from setuptools import setup, find_packages

exec(open("nyoka/metadata.py").read())

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
	long_description = f.read()


setup(
	name = "nyoka",
	version = __version__,
	description = "Python library for converting a large number of ML / DL models to PMML",
	long_description = long_description,
	long_description_content_type="text/markdown",
	author = "maintainer",
	author_email = "maintainer@nyoka.org",
	url = "https://github.com/softwareag/nyoka",
	license = __license__,
	classifiers = [
		"Development Status :: 5 - Production/Stable",
		"License :: OSI Approved :: Apache Software License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Intended Audience :: Developers",
		"Topic :: Scientific/Engineering :: Artificial Intelligence"
	],
	packages = find_packages(),
	python_requires= '>= 3.6',
	install_requires = [
		"lxml"
	]
)
