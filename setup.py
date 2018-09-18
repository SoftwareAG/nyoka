from setuptools import setup, find_packages

setup(
	name = "nyoka",
	version = "1.0.0",
	description = 'A Python library to export Machine Learning/ Deep Learning models into PMML',
	long_description = ''' Nyoka is a Python library for comprehensive support of the latest PMML standard plus extensions for data preprocessing, 
					  script execution and highly compacted representation of deep neural networks. Using Nyoka, Data Scientists can export a 
					  large number of Machine Learning and Deep Learning models from popular Python frameworks into PMML by either using any of 
					  the numerous included ready-to-use exporters or by creating their own exporter for specialized/individual model types by 
					  simply calling a sequence of constructors.''',
	long_description_content_type='text/plain',
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
