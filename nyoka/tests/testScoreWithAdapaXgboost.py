import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor

from nyoka import xgboost_to_pmml
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
    def setUpClass(cls):
        print("******* Unit Test for xgboost *******")
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["Species"] = iris.target
        df["Binary"] = numpy.array([i%2 for i in range(df.shape[0])])
        cls.X = df[iris.feature_names]
        cls.Y = df["Species"]
        cls.Y_bin = df["Binary"]
        cls.features = iris.feature_names
        cls.test_file = 'nyoka/tests/test.csv'
        cls.X.to_csv(cls.test_file,index=False)
        cls.adapa_utility = AdapaUtility()

    def test_01_xgb_classifier(self):
        print("\ntest 01 (xgb classifier with preprocessing) [multi-class]\n")
        model = XGBClassifier()
        pipeline_obj = Pipeline([
            ('scaler',MaxAbsScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(self.X,self.Y)
        file_name = "test01xgboost.pmml"
        xgboost_to_pmml(pipeline_obj, self.features, 'Species', file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, self.test_file)
        model_pred = pipeline_obj.predict(self.X)
        model_prob = pipeline_obj.predict_proba(self.X)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_02_xgb_regressor(self):
        print("\ntest 02 (xgb regressor without preprocessing)\n")
        model = XGBRegressor()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(self.X,self.Y)
        file_name = "test02xgboost.pmml"
        xgboost_to_pmml(pipeline_obj, self.features, 'Species', file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, self.test_file)
        model_pred = pipeline_obj.predict(self.X)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    def test_03_xgb_classifier(self):
        print("\ntest 03 (xgb classifier with preprocessing) [binary-class]\n")
        model = XGBClassifier()
        pipeline_obj = Pipeline([
            ('scaler',MinMaxScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(self.X,self.Y_bin)
        file_name = "test03xgboost.pmml"
        xgboost_to_pmml(pipeline_obj, self.features, 'Species', file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, self.test_file)
        model_pred = pipeline_obj.predict(self.X)
        model_prob = pipeline_obj.predict_proba(self.X)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    @classmethod
    def tearDownClass(cls):
        for i in range(1,4):
            f_name = "test"+str(i).zfill(2)+"xgboost.pmml"
            try:
                os.unlink(f_name)
            except:
                pass
        print("\n******* Finished *******\n")
     
if __name__ == '__main__':
    unittest.main(warnings='ignore')