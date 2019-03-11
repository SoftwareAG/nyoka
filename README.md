# Nyoka

[![Build Status](https://travis-ci.org/nyoka-pmml/nyoka.svg?branch=master)](https://travis-ci.org/nyoka-pmml/nyoka)
[![PyPI version](https://badge.fury.io/py/nyoka.svg)](https://badge.fury.io/py/nyoka)
[![license](https://img.shields.io/github/license/nyoka-pmml/nyoka.svg)](https://github.com/nyoka-pmml/nyoka/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://badge.fury.io/py/nyoka)

<img src="/docs/nyoka_logo.PNG" alt="nyoka_logo" height="240" style="float:right"/>

## Overview

Nyoka is a Python library for comprehensive support of the latest PMML standard plus extensions for data preprocessing, script execution and highly compacted representation of deep neural networks. Using Nyoka, Data Scientists can export a large number of Machine Learning and Deep Learning models from popular Python frameworks into PMML by either using any of the numerous included ready-to-use exporters or by creating their own exporter for specialized/individual model types by simply calling a sequence of constructors.

Besides about 500 Python classes which each cover a PMML tag and all constructor parameters/attributes as defined in the standard, Nyoka also provides an increasing number of convenience classes and functions that make the Data Scientist’s life easier for example by reading or writing any PMML file in one line of code from within your favorite Python environment.

Nyoka comes to you with the complete source code in Python, extended HTML documentation for the classes/functions, and a growing number of Jupyter Notebook tutorials that help you familiarize yourself with the way Nyoka supports you in using PMML as your favorite Data Science transport file format.


Read the documentation at [Nyoka Documentation](http://docs.nyoka.org).

## List of libraries and models supported by Nyoka :

### Scikit-Learn (version >= 1.19.1):
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

* Mobilenet
* VGG-16
* VGG-19
* Inception
* ResNet

### LightGBM (version 2.2.2):

* LGBMClassifier
* LGBMRegressor

### XGBoost (version 0.81):

* XGBClassifier
* XGBRegressor

### Statsmodels (version 0.9.0):

* ARIMA
* ExponentialSmoothing


## Prerequisites

* Python 3.6

## Dependencies

nyoka requires:

* lxml


## Installation

You can install nyoka using:

```
pip install nyoka
```
	
## Usage

### Nyoka to export scikit-learn models:

Exporting a Support Vector Classifier pipeline object into PMML

```python
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.svm import SVC

iris = datasets.load_iris()
irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
irisd['Species'] = iris.target

features = irisd.columns.drop('Species')
target = 'Species'

pipeline_obj = Pipeline([
    ('svm',SVC())
])

pipeline_obj.fit(irisd[features],irisd[target])


from nyoka import skl_to_pmml

skl_to_pmml(pipeline_obj,features,target,"svc_pmml.pmml")
```

Exporting a Random Forest Classifier (along with pre-processing) pipeline object into PMML

```python
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
irisd['Species'] = iris.target

features = irisd.columns.drop('Species')
target = 'Species'

pipeline_obj = Pipeline([
    ("mapping", DataFrameMapper([
    (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()) , 
    (['petal length (cm)', 'petal width (cm)'], Imputer())
    ])),
    ("rfc", RandomForestClassifier(n_estimators = 100))
])

pipeline_obj.fit(irisd[features], irisd[target])


from nyoka import skl_to_pmml

skl_to_pmml(pipeline_obj, features, target, "rf_pmml.pmml")
```

### Nyoka to export xgboost models:

Exporting a XGBoost model into PMML

```python
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

boston = datasets.load_boston()
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()

pipeline_obj = Pipeline([
    ("scaling", StandardScaler()),
    ("model", XGBRegressor())
])

pipeline_obj.fit(X, y)


from nyoka import xgboost_to_pmml

xgboost_to_pmml(pipeline_obj, boston.feature_names, 'target', "xgb_pmml.pmml")
```

### Nyoka to export lightGBM models:

Exporting a LGBM model into PMML

```python
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor,LGBMClassifier


iris = datasets.load_iris()
irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
irisd['Species'] = iris.target

features = irisd.columns.drop('Species')
target = 'Species'

pipeline_obj = Pipeline([
    ('lgbmc',LGBMClassifier())
])

pipeline_obj.fit(irisd[features],irisd[target])


from nyoka import lgb_to_pmml

lgb_to_pmml(pipeline_obj,features,target,"lgbmc_pmml.pmml")
```

### Nyoka to export keras models:

Exporting a Mobilenet model into PMML

```python
from keras import applications
from keras.layers import Flatten, Dense
from keras.models import Model

model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))

activType='sigmoid'
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation=activType)(x)
model_final = Model(inputs =model.input, outputs = predictions,name='predictions')

from nyoka import KerasToPmml
cnn_pmml = KerasToPmml(model_final,dataSet='image',predictedClasses=['cats','dogs'])

cnn_pmml.export(open('2classMBNet.pmml', "w"), 0)
```


## Uninstallation

```
pip uninstall nyoka
```

## Support

You can ask questions at:

*	[https://stackoverflow.com](https://stackoverflow.com) by tagging your questions with #pmml, #nyoka
*	You can also post bug reports in [GitHub issues](https://github.com/nyoka-pmml/nyoka/issues) 
 
