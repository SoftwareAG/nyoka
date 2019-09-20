import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import pandas as pd
from nyoka import ArimaToPMML
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
        print("******* Unit Test for Statsmodels *******")
        self.adapa_utility = AdapaUtility()

    def getData(self):
        # Non Seasonal Data
        data = [266,146,183,119,180,169,232,225,193,123,337,186,194,150,210,273,191,287,
                226,304,290,422,265,342,340,440,316,439,401,390,490,408,490,420,520,480]
        index = pd.DatetimeIndex(start='2016-01-01', end='2018-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'date_index'
        ts_data.name = 'cars_sold'
        ts_data = ts_data.astype('float64')
        return ts_data

    def test_01(self):
        ts_data = getData()
        f_name='arima201_c_car_sold.pmml'
        model = ARIMA(ts_data,order=(2,0,1))
        result = model.fit(trend = 'c', method = 'css')
        ArimaToPMML(result, f_name)

        model_name = self.adapaUtilities.upload_to_zserver(f_name)
        z_pred = self.adapaUtilities.score_single_record(model_name)
        model_pred = result.forecast()[0][0]
        self.assertEqual(model_pred, z_pred['predicted_cars_sold'])

        z_pred = self.adapa_utility.score_in_zserver(model_name, 'nyoka/tests/test_car_sold.csv','TS')
        model_pred = result.forecast(5)[0][-1]
        self.assertEqual(model_pred, z_pred)

    def test_02(self):
        data=pd.read_csv("nyoka/tests/JohnsonJohnsonWithDate.csv")
        data['index']=pd.to_datetime(data['index'], format='%Y-%m-%d')
        data.set_index(['index'], inplace=True)
        
        mod = SARIMAX(data,order=(1,0,0),seasonal_order=(1,0,0, 4))
        result = mod.fit()

        ArimaToPMML(result, 'jnj_seasonal_arima.pmml')
        model_name = self.adapaUtilities.upload_to_zserver('jnj_seasonal_arima.pmml')
        z_pred = self.adapaUtilities.score_single_record(model_name)
        model_pred = result.forecast()[0]
        self.assertEqual(model_pred, z_pred['predicted_value'])

        z_pred = self.adapa_utility.score_in_zserver(model_name, 'nyoka/tests/test_jnj.csv','TS')
        model_pred = result.forecast(5)[-1]
        self.assertEqual(model_pred, z_pred)


    @classmethod
    def tearDownClass(self):
        print("\n******* Finished *******\n")
     
if __name__ == '__main__':
    unittest.main(warnings='ignore')