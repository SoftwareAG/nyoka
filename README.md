
# Nyoka

[![Build Status](https://travis-ci.org/nyoka-pmml/nyoka.svg?branch=master)](https://travis-ci.org/nyoka-pmml/nyoka)
[![PyPI version](https://badge.fury.io/py/nyoka.svg)](https://pypi.org/project/nyoka/)
[![codecov](https://codecov.io/gh/nyoka-pmml/nyoka/branch/master/graph/badge.svg)](https://codecov.io/gh/nyoka-pmml/nyoka)
[![license](https://img.shields.io/github/license/nyoka-pmml/nyoka.svg)](https://github.com/nyoka-pmml/nyoka/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue)](https://pypi.org/project/nyoka/)

<img src="https://raw.githubusercontent.com/nyoka-pmml/nyoka/master/docs/nyoka_logo.PNG" alt="nyoka_logo" height="200" style="float:right"/>

## Overview

Nyoka is a Python library for comprehensive support of the latest PMML (PMML 4.4) standard. Using Nyoka, Data Scientists can export a large number of Machine Learning and Deep Learning models from popular Python frameworks into PMML by either using any of the numerous included ready-to-use exporters or by creating their own exporter for specialized/individual model types by simply calling a sequence of constructors.

Besides about 500 Python classes which each cover a PMML tag and all constructor parameters/attributes as defined in the standard, Nyoka also provides an increasing number of convenience classes and functions that make the Data Scientist’s life easier for example by reading or writing any PMML file in one line of code from within your favorite Python environment.

Nyoka comes to you with the complete source code in Python, extended HTML documentation for the classes/functions, and a growing number of Jupyter Notebook tutorials that help you familiarize yourself with the way Nyoka supports you in using PMML as your favorite Data Science transport file format.


Read the documentation at [Nyoka Documentation](http://docs.nyoka.org).

## List of libraries and models supported by Nyoka :

### Scikit-Learn (version <= 0.20.3):

#### Models -
* LinearRegression
* LogisticRegression
* RidgeClassifier
* SGDClassifier
* LinearDiscriminantAnalysis
* LinearSVC
* LinearSVR
* DecisionTreeClassifier
* DecisionTreeRegressor
* SVC
* SVR
* OneClassSVM
* GaussianNB
* RandomForestRegressor
* RandomForestClassifier
* GradientBoostingRegressor
* GradientBoostingClassifier
* IsolationForest
* MLPClassifier
* MLPRegressor
* KNNClassifier
* KNNRegressor
* KMeans

#### Pre-Processing -

* StandardScaler
* MinMaxScaler
* RobustScaler
* MaxAbsScaler
* TfidfVectorizer
* CountVectorizer
* LabelEncoder
* Imputer
* Binarizer
* PolynomialFeatures
* PCA
* LabelBinarizer
* OneHotEncoder
* CategoricalImputer

### Keras (version 2.2.4):

#### Models -
* Mobilenet
* VGG
* DenseNet
* Inception
* ResNet
* Xception
* Custom models

### Object Detection Model:
* Keras-RetinaNet

### LightGBM:

* LGBMClassifier
* LGBMRegressor


### XGBoost:

* XGBClassifier
* XGBRegressor

### Statsmodels:

* ARIMA (both old and new implementation)
* SARIMAX
* VARMAX
* ExponentialSmoothing

## Prerequisites

* Python >= 3.6, < 3.8

## Dependencies

nyoka requires:

* lxml


## Installation

You can install nyoka using:

```
pip install --upgrade nyoka
```
	
## Usage

Nyoka contains seperate exporters for each library, e.g., scikit-learn, keras, xgboost etc.

| library | exporter |
|--|--|
| **scikit-learn** | _skl_to_pmml_ |
| **xgboost** | _xgboost_to_pmml_ |
| **lightgbm** | _lgbm_to_pmml_ |
| **keras** | _KerasToPmml_ |
| **statsmodels** | _StatsmodelsToPmml & ExponentialSmoothingToPmml_ |
| **retinanet** | _RetinanetToPmml_ |

The main module of __Nyoka__ is `nyoka`. To use it for your model, you need to import the specific exporter from nyoka as -

```python
from nyoka import skl_to_pmml, lgb_to_pmml #... so on
```
#### Note -
 - If scikit-learn, xgboost and lightgbm model is used then the model should be used inside sklearn's Pipeline.
	The workflow is as follows -
	* Create scikit-learn's `Pipeline` object and populate it with any preprocessing steps and the model object.
	* Call `Pipeline.fit(X,y)` method to train the model.
	* Use the specific exporter and pass the pipeline object, feature names of the training dataset, target name and expected name of the PMML to the exporter function. If target name is not given default value `target` is used. Similarly, for pmml name, default value `from_sklearn.pmml`/`from_xgboost.pmml`/`from_lighgbm.pmml` is used. 
 - For Keras, RetinaNet and Statsmodels, the fitted model needs to be passed to the exporter.
 
 ___Demo can be found in https://github.com/nyoka-pmml/nyoka/tree/master/examples___


## More in Nyoka
Nyoka contains one submodule called `preprocessing`. This module contains preprocessing classes implemented by Nyoka. Currently there is only one preprocessing class, which is `Lag`.

#### What is Lag? When to use it?
>Lag is a preprocessing class implemented by Nyoka. When used inside scikit-learn's pipeline, it simply applies  an `aggregation` function for the given features of the dataset by combining `value` number of previous records. It takes two arguments- aggregation and value.
>
> The valid `aggregation` functions are -
"min",  "max",  "sum",  "avg",  "median",  "product" and "stddev".
>
To use __Lag__ -
* Import it from nyoka as `from nyoka.preprocessing import Lag`
* Create an instance of Lag as `Lag(aggregation="sum", value=5)`
	* This means, take 5 previous values for the given fields and perform summation.
* Use this object inside scikit-learn's pipeline to train.

## Uninstallation

```
pip uninstall nyoka
```

## Support

You can ask questions at:

*	[Stack Overflow](https://stackoverflow.com) by tagging your questions with #pmml, #nyoka
*	You can also post bug reports in [GitHub issues](https://github.com/nyoka-pmml/nyoka/issues) 
 
