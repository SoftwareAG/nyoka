
# Nyoka

[![Build Status](https://travis-ci.org/nyoka-pmml/nyoka.svg?branch=master)](https://travis-ci.org/nyoka-pmml/nyoka)
[![PyPI version](https://badge.fury.io/py/nyoka.svg)](https://badge.fury.io/py/nyoka)
[![codecov](https://codecov.io/gh/nyoka-pmml/nyoka/branch/master/graph/badge.svg)](https://codecov.io/gh/nyoka-pmml/nyoka)
[![license](https://img.shields.io/github/license/nyoka-pmml/nyoka.svg)](https://github.com/nyoka-pmml/nyoka/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://badge.fury.io/py/nyoka)

<img src="https://raw.githubusercontent.com/nyoka-pmml/nyoka/master/docs/nyoka_logo.PNG" alt="nyoka_logo" height="200" style="float:right"/>

## Overview

Nyoka is a Python library for comprehensive support of the latest PMML (PMML 4.4) standard. Using Nyoka, Data Scientists can export a large number of Machine Learning and Deep Learning models from popular Python frameworks into PMML by either using any of the numerous included ready-to-use exporters or by creating their own exporter for specialized/individual model types by simply calling a sequence of constructors.

Besides about 500 Python classes which each cover a PMML tag and all constructor parameters/attributes as defined in the standard, Nyoka also provides an increasing number of convenience classes and functions that make the Data Scientist’s life easier for example by reading or writing any PMML file in one line of code from within your favorite Python environment.

Nyoka comes to you with the complete source code in Python, extended HTML documentation for the classes/functions, and a growing number of Jupyter Notebook tutorials that help you familiarize yourself with the way Nyoka supports you in using PMML as your favorite Data Science transport file format.


Read the documentation at [Nyoka Documentation](http://docs.nyoka.org).

## List of libraries and models supported by Nyoka :

### Scikit-Learn (version <= 0.20.3):
<details><summary>Click to expand!</summary>

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
</Details>

### Keras (version 2.2.4):
<details><summary>Click to expand!</summary>

#### Models -
* Mobilenet
* VGG
* DenseNet
* Inception
* ResNet
* Xception
</details>

### Object Detection Model:
* Keras-RetinaNet

### LightGBM:
<details><summary>Click to expand!</summary>

#### Models -
* LGBMClassifier
* LGBMRegressor
</details>

### XGBoost:
<details><summary>Click to expand!</summary>

#### Models -
* XGBClassifier
* XGBRegressor
</details>

### Statsmodels:
<details><summary>Click to expand!</summary>

#### Models -
* ARIMA
* SARIMAX
* ExponentialSmoothing
</details>

## Prerequisites

* Python 3.6

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
| **statsmodels** | _ArimaToPmml & ExponentialSmoothingToPmml_ |
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
 - For Keras and Statsmodels, the fitted model needs to be passed to the exporter.
 
 ___Demo is provided below___
### Nyoka to export scikit-learn models:

>Exporting a Support Vector Classifier pipeline object into PMML

```python
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()
irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
irisd['Species'] = iris.target
features = irisd.columns.drop('Species')
target = 'Species'

pipeline_obj = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',SVC())
])
pipeline_obj.fit(irisd[features],irisd[target])

from nyoka import skl_to_pmml
skl_to_pmml(pipeline_obj,features,target,"svc_pmml.pmml")
```

### Nyoka to export xgboost models:

>Exporting a XGBoost model into PMML

```python
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

>Exporting a LGBM model into PMML

```python
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


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

>Exporting a Mobilenet model into PMML

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

>Exporting user given python script with keras model

```python
from keras import applications
from keras.layers import Flatten, Dense
from keras.models import Model
model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
script_content = open("preprocess.py",'r').read()
pmml_obj=KerasToPmml(model_final,
            dataSet='image',
            predictedClasses=['cat','dog'],
            script_args = {
                "content" : script_content,
                "def_name" : "getBase64EncodedString",
                "return_type" : "string",
                "encode":True
            }
        )
pmml_obj.export(open("script_with_keras_encoded.pmml",'w'),0)
```

### Nyoka to export object detection model

>Exporting RetinaNet to PMML
```python
from keras_retinanet.models import load_model
from nyoka import RetinanetToPmml
model = load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')
backbone = 'resnet'
RetinanetToPmml(
    model,
    input_shape=(224,224,3),
    input_format="image",
    backbone_name=backbone,
    pmml_file_name="retinanet_with_coco_.pmml"
)
```

### Nyoka to export statsmodels model
>Exporting Non Seasonal ARIMA to PMML
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from nyoka import ArimaToPMML

def parser(x):
    return pd.datetime.strptime(x,'%Y-%m')

sales_data = pd.read_csv('sales-cars.csv', index_col=0, parse_dates = [0], date_parser = parser)
model = ARIMA(sales_data, order = (9, 2, 0))
result = model.fit()

pmml_f_name = 'non_seasonal_car_sales.pmml'
ArimaToPMML(results_obj = result,pmml_file_name = pmml_f_name)
```

>Exporting Seasonal ARIMA to PMML
```python
import pandas as pd
from nyoka import ArimaToPMML
from statsmodels.tsa.statespace.sarimax import SARIMAX
data=pd.read_csv("JohnsonJohnsonWithDate.csv")
data['index']=pd.to_datetime(data['index'], format='%Y-%m-%d')
data.set_index(['index'], inplace=True)

mod = SARIMAX(data,order=(1,0,0),seasonal_order=(1,0,0, 4))
result = mod.fit()

ArimaToPMML(results, 'jnj_seasonal_arima.pmml')
```

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
 
