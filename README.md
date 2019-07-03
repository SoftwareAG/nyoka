
# Nyoka

[![license](https://img.shields.io/github/license/nyoka-pmml/nyoka.svg)](https://github.com/nyoka-pmml/nyoka/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://badge.fury.io/py/nyoka)

<img src="https://raw.githubusercontent.com/nyoka-pmml/nyoka/master/docs/nyoka_logo.PNG" alt="nyoka_logo" height="200" style="float:right"/>

## Overview

Nyoka is a Python library for comprehensive support of the latest PMML (PMML 4.4) standard plus plus extensions for data preprocessing, script execution and highly compacted representation of deep neural networks. Using Nyoka, Data Scientists can export a large number of Machine Learning and Deep Learning models from popular Python frameworks into PMML by either using any of the numerous included ready-to-use exporters or by creating their own exporter for specialized/individual model types by simply calling a sequence of constructors.

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
* VGG-16
* VGG-19
* Inception
* ResNet
</details>

### LightGBM (version 2.2.2):
<details><summary>Click to expand!</summary>

#### Models -
* LGBMClassifier
* LGBMRegressor
</details>

### XGBoost (version 0.81):
<details><summary>Click to expand!</summary>

#### Models -
* XGBClassifier
* XGBRegressor
</details>

### Statsmodels (version 0.9.0):
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
pip install git+https://github.com/nyoka-pmml/nyoka.git@44Ext 
```
	
## Usage

Nyoka contains seperate exporters for each library, e.g., scikit-learn, keras, xgboost etc.

| library | exporter |
|--|--|
| **scikit-learn** | _skl_to_pmml_ |
| **xgboost** | _xgboost_to_pmml_ |
| **lightgbm** | _lgbm_to_pmml_ |
| **keras** | _KerasToPmml_ |
| **statsmodels** | _ArimaToPmml & ExponentialSmoothingToPmml_

The main module of __Nyoka__ is `nyoka`. To use it for your model, you need to import the specific exporter from nyoka as -

```python
from nyoka import model_to_pmml
```
 
 ___Demo is provided below___
### Nyoka to export scikit-learn models:

>Exporting a Random Forest Classifier object into PMML with Pipeline and custome preprocessing steps

```python

def script1():
    r3 = r1+r2
    
def script2():
    r6 = r1+r2+r3-r4

import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nyoka import model_to_pmml

iris = datasets.load_iris()
irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
irisd['Species'] = iris.target

features = irisd.columns.drop('Species')
target = 'Species'

# Build a pipeline of pre-processing steps
pipelineOnly = Pipeline([
    ('Scaling', StandardScaler()), 
    ('Imputing', Imputer())
])

pipelineOnly.fit(irisd[features])
Xdata = pipelineOnly.transform(irisd[features])

# Build a Random Forest classifier model
rfObj = RandomForestClassifier()
rfObj.fit(Xdata,irisd['Species'])

# Export into PMML
toExportDict={
    'model1':{
        'hyperparameters':None,
        'preProcessingScript':{'scripts':[script1,script2], 'scriptpurpose':['train','score']},
        'pipelineObj':pipelineOnly,
        'modelObj':rfObj,
        'featuresUsed':features,
        'targetName':'Species',
        'postProcessingScript':{'scripts':[script1], 'scriptpurpose':['postprocess']},
        'taskType': 'trainAndscore'
    }
}

model_to_pmml(toExportDict, pmml_f_name="sklearnppOnly.pmml")
```

### Nyoka to export lightGBM Train models:

>Exporting a LGBM Train Model into PMML

```python
from sklearn import datasets
import pandas as pd
import lightgbm as lgb


iris = datasets.load_iris()
irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
irisd['Species'] = iris.target
features = irisd.columns.drop('Species')
target = 'Species'

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(irisd[features], irisd[target], test_size=0.3, random_state=0) 

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train , free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':3,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 20
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                verbose_eval = True
               )

from nyoka import model_to_pmml

toExportDict={
    'model1':{
        'hyperparameters':params,
        'preProcessingScript':None,
        'pipelineObj':None,
        'modelObj':gbm,
        'featuresUsed':features,
        'targetName':target,
        'postProcessingScript':None,
        'taskType': 'trainAndscore'
    }
}

model_to_pmml(toExportDict, pmml_f_name="LGBM_Train_API_Example.pmml")

```

### Nyoka to export keras models:

>Exporting a Keras model into PMML

```python

def script1():
    r3 = r1+r2
    
def script2():
    r6 = r1+r2+r3-r4

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128, verbose=0)

toExportDict={
    'model1':{
        'hyperparameters':None,
        'preProcessingScript':{'scripts':[script1,script2], 'scriptpurpose':['train','score']},
        'pipelineObj':pipelineOnly,
        'modelObj':model,
        'featuresUsed':features,
        'targetName':None,
        'postProcessingScript':{'scripts':[script1], 'scriptpurpose':['postprocess']},
        'taskType': 'trainAndscore'
    }
}

model_to_pmml(toExportDict, pmml_f_name="KerasppOnly.pmml")
```

### Nyoka to export multiple models:

>Exporting multiple models at once into PMML

```python
def script1():
    r3 = r1+r2
    
def script2():
    r6 = r1+r2+r3-r4

# First model
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nyoka import model_to_pmml

iris = datasets.load_iris()
irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
irisd['Species'] = iris.target

features = irisd.columns.drop('Species')
target = 'Species'

# Build a pipeline of pre-processings
pipelineOnly = Pipeline([
    ('Scaling', StandardScaler()), 
    ('Imputing', Imputer())
])

pipelineOnly.fit(irisd[features])
Xdata = pipelineOnly.transform(irisd[features])

# Build a Random Forest Classifier model
rfObj = RandomForestClassifier()
rfObj.fit(Xdata,irisd['Species'])


# Second model
import sklearn.preprocessing
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD 
from nyoka import model_to_pmml

x = preprocessingScript()

data = []
for x,y in zip(x['temperature'].values, x['pressure'].values):
    data+= [x,y]
    
data = np.asarray(data)
data = data.reshape(500, 10, 2)

#create labels
labels = np.random.randint(4, size=500)
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(labels)+1))
labels = label_binarizer.transform(labels)

verbose, epochs, batch_size = 1, 5, 32

# Build a LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(10,2)))
# model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data, labels, epochs=epochs, batch_size=32, verbose=verbose)

# Export multiple models into PMML using Nyoka

toExportDict={
    'model1':{
        'hyperparameters':None,
        'preProcessingScript':{'scripts':[script1,script2], 'scriptpurpose':['train','score']},
        'pipelineObj':pipelineOnly,
        'modelObj':rfObj,
        'featuresUsed':features,
        'targetName':'Species',
        'postProcessingScript':{'scripts':[script1], 'scriptpurpose':['postprocess']},
        'taskType': 'trainAndscore'
    },
    'model2':{
        'hyperparameters':None,
        'preProcessingScript':{'scripts':[script1], 'scriptpurpose':['train']},
        'pipelineObj':None,
        'modelObj':model,
        'featuresUsed':None,
        'targetName':None,
        'postProcessingScript':{'scripts':[script1], 'scriptpurpose':['postprocess']},
        'taskType': 'score'
    },
}

model_to_pmml(toExportDict, pmml_f_name="MultipleModels.pmml")

```


### Nyoka to export statsmodels model
>Exporting ARIMA to PMML
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

## Uninstallation

```
pip uninstall nyoka
```

## Support

You can ask questions at:

*	[Stack Overflow](https://stackoverflow.com) by tagging your questions with #pmml, #nyoka
*	You can also post bug reports in [GitHub issues](https://github.com/nyoka-pmml/nyoka/issues) 
 
