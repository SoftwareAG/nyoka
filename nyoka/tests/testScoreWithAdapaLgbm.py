import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

from nyoka import lgb_to_pmml
from nyoka import PMML44 as pml
import unittest
import requests
import json
from requests.auth import HTTPBasicAuth
import ast
import numpy
from adapaUtilities import AdapaUtility

class TestCases(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("******* Unit Test for lightgbm *******")
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["Species"] = iris.target
        df["Binary"] = numpy.array([i%2 for i in range(df.shape[0])])
        self.X = df[iris.feature_names]
        self.Y = df["Species"]
        self.Y_bin = df["Binary"]
        self.features = iris.feature_names
        self.test_file = 'nyoka/tests/test.csv'
        self.X.to_csv(self.test_file,index=False)
        self.adapa_utility = AdapaUtility()

    
    def test_01_lgbm_classifier(self):
        print("\ntest 01 (lgbm classifier with preprocessing) [binary-class]\n")
        model = LGBMClassifier()
        pipeline_obj = Pipeline([
            ('scaler',MinMaxScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(self.X,self.Y_bin)
        file_name = "test01lgbm.pmml"
        lgb_to_pmml(pipeline_obj, self.features, 'Species', file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, self.test_file)
        model_pred = pipeline_obj.predict(self.X)
        model_prob = pipeline_obj.predict_proba(self.X)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_02_lgbm_classifier(self):
        print("\ntest 02 (lgbm classifier with preprocessing) [multi-class]\n")
        model = LGBMClassifier()
        pipeline_obj = Pipeline([
            ('scaler',MaxAbsScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(self.X,self.Y)
        file_name = "test02lgbm.pmml"
        lgb_to_pmml(pipeline_obj, self.features, 'Species', file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, self.test_file)
        model_pred = pipeline_obj.predict(self.X)
        model_prob = pipeline_obj.predict_proba(self.X)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_03_lgbm_regressor(self):
        print("\ntest 03 (lgbm regressor without preprocessing)\n")
        model = LGBMRegressor()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(self.X,self.Y)
        file_name = "test03lgbm.pmml"
        lgb_to_pmml(pipeline_obj, self.features, 'Species', file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, self.test_file)
        model_pred = pipeline_obj.predict(self.X)
        predictions = numpy.array(predictions)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)


    def test_04_lgbm_regressor(self):
        print("\ntest 04 (lgbm regressor with preprocessing)\n")
        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in ('mpg')]
        target_name='mpg'
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        pd.DataFrame(data=x_test, columns=feature_names).to_csv("test.csv",index=False)
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', CountVectorizer()),
                (['displacement'],[StandardScaler()]) 
            ])),
            ('lgbmr',LGBMRegressor())
        ])
        pipeline_obj.fit(x_train,y_train)
        file_name = "test04lgbm.pmml"
        lgb_to_pmml(pipeline_obj, feature_names, 'mpg', file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, "test.csv")
        predictions = numpy.array(predictions)
        model_pred = pipeline_obj.predict(x_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    @classmethod
    def tearDownClass(self):
        print("\n******* Finished *******\n")
     
# if __name__ == '__main__':
#     unittest.main(warnings='ignore')