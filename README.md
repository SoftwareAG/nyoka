# Nyoka

[![Test Master Branch](https://github.com/SoftwareAG/nyoka/actions/workflows/test-master.yml/badge.svg?branch=master&event=push)](https://github.com/SoftwareAG/nyoka/actions/workflows/test-master.yml)
[![PyPI version](https://badge.fury.io/py/nyoka.svg)](https://pypi.org/project/nyoka/)
[![codecov](https://codecov.io/gh/SoftwareAG/nyoka/branch/master/graph/badge.svg)](https://codecov.io/gh/SoftwareAG/nyoka)
[![license](https://img.shields.io/github/license/softwareag/nyoka.svg)](https://github.com/softwareag/nyoka/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%2B-blue)](https://pypi.org/project/nyoka/)
<img  src="https://raw.githubusercontent.com/softwareag/nyoka/master/docs/nyoka_logo.PNG"  alt="nyoka_logo"  height="200"  style="float:right"/>

## Overview

Nyoka is a Python library for comprehensive support of the latest PMML (PMML 4.4) standard. Using Nyoka, Data Scientists can export a large number of Machine Learning models from popular Python frameworks into PMML by either using any of the numerous included ready-to-use exporters or by creating their own exporter for specialized/individual model types by simply calling a sequence of constructors.

Besides about 500 Python classes which each cover a PMML tag and all constructor parameters/attributes as defined in the standard, Nyoka also provides an increasing number of convenience classes and functions that make the Data Scientist’s life easier for example by reading or writing any PMML file in one line of code from within your favorite Python environment.

Nyoka comes to you with the complete source code in Python, extended HTML documentation for the classes/functions, and a growing number of Jupyter Notebook tutorials that help you familiarize yourself with the way Nyoka supports you in using PMML as your favorite Data Science transport file format.

Read the documentation at **[Nyoka Documentation](https://softwareag.github.io/nyoka/)**.

## List of libraries and models supported by Nyoka :

### Scikit-Learn (version <= 0.23.1):

#### Models -

*  [`linear_model.LinearRegression`](https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
*  [`linear_model.LogisticRegression`](https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
*  [`linear_model.RidgeClassifier`](https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier)
*  [`linear_model.SGDClassifier`](https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)
*  [`discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org/0.20/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
*  [`tree.DecisionTreeClassifier`](https://scikit-learn.org/0.20/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
*  [`tree.DecisionTreeRegressor`](https://scikit-learn.org/0.20/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)
*  [`svm.SVC`](https://scikit-learn.org/0.20/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
*  [`svm.SVR`](https://scikit-learn.org/0.20/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)
*  [`svm.LinearSVC`](https://scikit-learn.org/0.20/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
*  [`svm.LinearSVR`](https://scikit-learn.org/0.20/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)
*  [`svm.OneClassSVM`](https://scikit-learn.org/0.20/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM)
*  [`naive_bayes.GaussianNB`](https://scikit-learn.org/0.20/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)
*  [`ensemble.RandomForestRegressor`](https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)
*  [`ensemble.RandomForestClassifier`](https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
*  [`ensemble.GradientBoostingRegressor`](https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)
*  [`ensemble.GradientBoostingClassifier`](https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)
*  [`ensemble.IsolationForest`](https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest)
*  [`neural_network.MLPClassifier`](https://scikit-learn.org/0.20/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
*  [`neural_network.MLPRegressor`](https://scikit-learn.org/0.20/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)
*  [`neighbors.KNeighborsClassifier`](https://scikit-learn.org/0.20/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
*  [`neighbors.KNeighborsRegressor` ](https://scikit-learn.org/0.20/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)
*  [`cluster.KMeans`](https://scikit-learn.org/0.20/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)


#### Pre-Processing -


*  [`preprocessing.StandardScaler`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
*  [`preprocessing.MinMaxScaler`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)
*  [`preprocessing.RobustScaler`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler)
*  [`preprocessing.MaxAbsScaler`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler)
*  [`preprocessing.LabelEncoder`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder)
*  [`preprocessing.Imputer`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer)
*  [`preprocessing.Binarizer`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer)
*  [`preprocessing.PolynomialFeatures`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures)
*  [`preprocessing.LabelBinarizer`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer)
*  [`preprocessing.OneHotEncoder`](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)
*  [`feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org/0.20/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
*  [`feature_extraction.text.CountVectorizer`](https://scikit-learn.org/0.20/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)
*  [`decomposition.PCA`](https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
*  [`sklearn_pandas.CategoricalImputer`](https://github.com/scikit-learn-contrib/sklearn-pandas/blob/master/sklearn_pandas/categorical_imputer.py#L21) ( From _[sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas)_ library )
  

### LightGBM:

  
*  [`LGBMClassifier`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
*  [`LGBMRegressor`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)


### XGBoost (version <= 1.5.2):


*  [`XGBClassifier`](https://xgboost.readthedocs.io/en/release_1.5.0/python/python_api.html#module-xgboost.sklearn)
*  [`XGBRegressor`](https://xgboost.readthedocs.io/en/release_1.5.0/python/python_api.html#module-xgboost.sklearn)


### Statsmodels (version <= 0.11.1):


*  [`tsa.arima_model.ARIMA`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima_model.py#L1026)
*  [`tsa.arima.model.ARIMA`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima/model.py#L26) _(Extension of SARIMAX)_
*  [`tsa.statespace.SARIMAX`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/statespace/sarimax.py#L31)
*  [`tsa.statespace.VARMAX`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/statespace/varmax.py#L33)
*  [`tsa.statespace.ExponentialSmoothing`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/statespace/exponential_smoothing.py#L31)
  

## Prerequisites

* Python >= 3.6

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
| **statsmodels** | _StatsmodelsToPmml & ExponentialSmoothingToPmml_ |


The main module of __Nyoka__ is `nyoka`. To use it for your model, you need to import the specific exporter from nyoka as -

```python
from nyoka import skl_to_pmml, lgb_to_pmml #... so on
```

#### Note - If scikit-learn, xgboost and lightgbm model is used then the model should be used inside sklearn's Pipeline.

The workflow is as follows (For example, a Decision Tree Classifier with StandardScaler) -

* Create scikit-learn's `Pipeline` object and populate it with any pre-processing steps and the model object. 
	```python
	from sklearn.pipeline import Pipeline
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.preprocessing import StandardScaler
	pipeline_obj = Pipeline([
			("scaler",StandardScaler()),
			("model",DecisionTreeClassifier())
	])
	```

* Call `Pipeline.fit(X,y)` method to train the model.
	```python
	from sklearn.dataset import load_iris
	iris_data = load_iris()
	X = iris_data.data
	y = iris_data.target
	features = iris_data.feature_names
	pipeline_obj.fit(X,y)
	```
  
* Use the specific exporter and pass the pipeline object, feature names of the training dataset, target name and expected name of the PMML to the exporter function. If target name is not given default value `target` is used. Similarly, for pmml name, default value `from_sklearn.pmml`/`from_xgboost.pmml`/`from_lighgbm.pmml` is used.
	```python
	from nyoka import skl_to_pmml
	skl_to_pmml(pipeline=pipeline_obj,col_names=features,target_name="species",pmml_f_name="decision_tree.pmml")
	```


#### For Statsmodels, pipeline is not required. The fitted model needs to be passed to the exporter.

```python
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from nyoka import StatsmodelsToPmml
sales_data = pd.read_csv('sales-cars.csv', index_col=0, parse_dates = True)
model = ARIMA(sales_data, order = (4, 1, 2))
result = model.fit()
StatsmodelsToPmml(result,"Sales_cars_ARIMA.pmml")
```

## Examples 

Example jupyter notebooks can be found in [`nyoka/examples`](https://github.com/softwareag/nyoka/tree/master/examples). These files contain code to showcase how to use different exporters.

* Exporting `scikit-learn` models into PMML
	* [SVM](https://github.com/softwareag/nyoka/blob/master/examples/skl/1_SVM.ipynb)
	* [KNeighbors](https://github.com/softwareag/nyoka/blob/master/examples/skl/2_K-NN_With_Scaling.ipynb)
	* [Random Forest](https://github.com/softwareag/nyoka/blob/master/examples/skl/3_RF_With_pre-processing.ipynb)
	* [Gardient Boosting](https://github.com/softwareag/nyoka/blob/master/examples/skl/4_GB_With_pre-processing.ipynb)
	* [Decision Tree](https://github.com/softwareag/nyoka/blob/master/examples/skl/5_Decision_Tree_With_Tf-Idf.ipynb)
	* [Isolation Forest](https://github.com/softwareag/nyoka/blob/master/examples/skl/6_IsolationForest_model_to_PMML.ipynb)
	* [OneClassSVM](https://github.com/softwareag/nyoka/blob/master/examples/skl/7_OneClassSVM_Model_to_PMML.ipynb)
	* [LinearSVC](https://github.com/softwareag/nyoka/blob/master/examples/skl/8_LinearSVC_with_TfidfVectorizer.ipynb)

* Exporting `XGBoost` model into PMML
	* [XGBoost 1](https://github.com/softwareag/nyoka/blob/master/examples/xgboost/1_xgboost.ipynb)
	* [XGBoost 2](https://github.com/softwareag/nyoka/blob/master/examples/xgboost/2_xgboost_With_Scaling.ipynb)
	* [XGBoost 3](https://github.com/softwareag/nyoka/blob/master/examples/xgboost/3_xgboost_With_PreProcess%20.ipynb)

* Exporting `LightGBM` model into PMML
	* [LightGBM 1](https://github.com/softwareag/nyoka/blob/master/examples/lgbm/1_lgbm.ipynb)
	* [LightGBM 2](https://github.com/softwareag/nyoka/blob/master/examples/lgbm/2_lgbm_With_Scaling.ipynb)
	* [LightGBM 3](https://github.com/softwareag/nyoka/blob/master/examples/lgbm/3_lgbm_With_PreProcess%20.ipynb)

* Exporting `statsmodels` model into PMML
	* [Non-Seasonal ARIMA](https://github.com/softwareag/nyoka/blob/master/examples/statsmodels/arima/Non-Seasonal%20ARIMA.ipynb)
	* [Seasonal ARIMA](https://github.com/softwareag/nyoka/blob/master/examples/statsmodels/arima/Seasonal%20ARIMA.ipynb)
	* [Vector ARMA (for multi-variate time series)](https://github.com/softwareag/nyoka/blob/master/examples/statsmodels/arima/VARMAX.ipynb)
	* [Exponential Smoothing](https://github.com/softwareag/nyoka/blob/master/examples/statsmodels/exponential_smoothing/exponential_smoothing.ipynb)
  
## Nyoka Submodules

Nyoka contains one submodule called `preprocessing`. This module contains preprocessing classes implemented by Nyoka. Currently there is only one preprocessing class, which is `Lag`.

#### What is Lag? When to use it?


>Lag is a preprocessing class implemented by Nyoka. When used inside scikit-learn's pipeline, it simply applies an `aggregation` function for the given features of the dataset by combining `value` number of previous records. It takes two arguments- aggregation and value.

>

> The valid `aggregation` functions are -
> "min", "max", "sum", "avg", "median", "product" and "stddev".


To use __Lag__ -

* Import it from nyoka -
  ```python
	from nyoka.preprocessing import Lag
  ```
* Create an instance of Lag - 
  ```python
	lag_obj = Lag(aggregation="sum", value=5)
	'''
	This means taking previous 5 values and perform `sum`. When used inside pipeline, this will be applied to all the columns.
	If used inside DataFrameMapper, the it will be applied to only those columns which are inside DataFrameMapper.
	'''
  ```
* Use this object inside scikit-learn's pipeline to train.
  ```python
	from sklearn.pipeline import Pipeline
	from sklearn.tree import DecisionTreeClassifier
	from nyoka.preprocessing import Lag
	pipeline_obj = Pipeline([
		("lag",Lag(aggregation="sum",value=5)),
		("model",DecisionTreeClassifier())
	])
  ```

## Uninstallation

```
pip uninstall nyoka
```

## Support

You can ask questions at:

*  [Stack Overflow](https://stackoverflow.com) by tagging your questions with #pmml, #nyoka
* You can also post bug reports in [GitHub issues](https://github.com/softwareag/nyoka/issues)

-----

Please note that this project is released with a [Contributor Code of
Conduct](https://github.com/SoftwareAG/nyoka/blob/master/.github/CODE_OF_CONDUCT.md).
By contributing to this project, you agree to abide by its terms.

These tools are provided as-is and without warranty or support. They do
not constitute part of the Software AG product suite. Users are free to
use, fork and modify them, subject to the license agreement. While
Software AG welcomes contributions, we cannot guarantee to include every
contribution in the master project.
