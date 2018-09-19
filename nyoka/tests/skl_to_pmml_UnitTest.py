import os
import unittest
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from nyoka import skl_to_pmml


class TestMethods(unittest.TestCase):

    
    def test_SVC(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('svm',SVC())
        ])

        pipeline_obj.fit(irisd[features],irisd[target])

        skl_to_pmml(pipeline_obj,features,target,"svc_pmml.pmml")

        self.assertEqual(os.path.isfile("svc_pmml.pmml"),True)

    
    def test_RF(self):
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

        skl_to_pmml(pipeline_obj, features, target, "rf_pmml.pmml")

        self.assertEqual(os.path.isfile("rf_pmml.pmml"),True)


if __name__=='__main__':
    unittest.main(warnings='ignore')







