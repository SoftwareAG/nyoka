import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
from nyoka import StatsmodelsToPmml
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA as StateSpaceARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
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
        print("******* Unit Test for Statsmodels *******")
        cls.adapa_utility = AdapaUtility()

    def getData(self):
        # Non Seasonal Data
        data = [266,146,183,119,180,169,232,225,193,123,337,186,194,150,210,273,191,287,
                226,304,290,422,265,342,340,440,316,439,401,390,490,408,490,420,520,480]
        index = pd.date_range(start='2016-01-01', end='2018-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'date_index'
        ts_data.name = 'cars_sold'
        ts_data = ts_data.astype('float64')
        return ts_data

    def getMultiDimensionalData(self):
        data = pd.read_csv("nyoka/tests/SanDiegoWeather.csv",parse_dates=True,index_col=0)
        return data

    def test_01(self):
        ts_data = self.getData()
        f_name='arima201_c_car_sold.pmml'
        model = ARIMA(ts_data,order=(2,0,1))
        result = model.fit(trend = 'c', method = 'css')
        StatsmodelsToPmml(result, f_name, conf_int=[95])

        model_name = self.adapa_utility.upload_to_zserver(f_name)
        z_pred = self.adapa_utility.score_in_zserver(model_name, {'h':5},'TS')

        z_forecasts = np.array(list(z_pred['outputs'][0]['predicted_'+ts_data.squeeze().name].values()))
        model_forecasts = result.forecast(5)[0]

        z_conf_int_95_upper = np.array(list(z_pred['outputs'][0]['conf_int_95_upper_'+ts_data.squeeze().name].values()))
        model_conf_int_95_upper = result.forecast(5)[-1][:,-1]

        z_conf_int_95_lower = np.array(list(z_pred['outputs'][0]['conf_int_95_lower_'+ts_data.squeeze().name].values()))
        model_conf_int_95_lower = result.forecast(5)[-1][:,0]

        self.assertEqual(np.allclose(z_forecasts, model_forecasts),True)
        self.assertEqual(np.allclose(z_conf_int_95_upper, model_conf_int_95_upper),True)
        self.assertEqual(np.allclose(z_conf_int_95_lower, model_conf_int_95_lower),True)


    def test_02(self):
        data=pd.read_csv("nyoka/tests/JohnsonJohnsonWithDate.csv")
        data['index']=pd.to_datetime(data['index'], format='%Y-%m-%d')
        data.set_index(['index'], inplace=True)
        
        mod = SARIMAX(data,order=(1,1,1),seasonal_order=(1,1,1, 4))
        result = mod.fit(disp=False)

        StatsmodelsToPmml(result, 'jnj_seasonal_arima.pmml',conf_int=[95])
        model_name = self.adapa_utility.upload_to_zserver('jnj_seasonal_arima.pmml')
        
        z_pred = self.adapa_utility.score_in_zserver(model_name, {'h':5},'TS')
        forecasts=result.get_forecast(5)

        z_forecasts = list(z_pred['outputs'][0]['predicted_'+data.squeeze().name].values())
        model_forecasts = forecasts.predicted_mean.values.tolist()

        z_conf_int_95_upper = list(z_pred['outputs'][0]['conf_int_95_upper_'+data.squeeze().name].values())
        model_conf_int_95_upper = forecasts.conf_int()['upper '+data.squeeze().name].tolist()

        z_conf_int_95_lower = list(z_pred['outputs'][0]['conf_int_95_lower_'+data.squeeze().name].values())
        model_conf_int_95_lower = forecasts.conf_int()['lower '+data.squeeze().name].tolist()

        self.assertEqual(np.allclose(z_forecasts,model_forecasts),True)
        self.assertEqual(np.allclose(z_conf_int_95_upper, model_conf_int_95_upper),True)
        self.assertEqual(np.allclose(z_conf_int_95_lower, model_conf_int_95_lower),True)
        

    def test_03(self):
        ts_data = self.getData()
        f_name='arima212_c_car_sold.pmml'
        model = StateSpaceARIMA(ts_data,order=(2,1,2),trend = 'c')
        result = model.fit()
        StatsmodelsToPmml(result, f_name, conf_int=[95])

        model_name = self.adapa_utility.upload_to_zserver(f_name)
        z_pred = self.adapa_utility.score_in_zserver(model_name, {'h':5},'TS')
        forecasts=result.get_forecast(5)

        z_forecasts = list(z_pred['outputs'][0]['predicted_'+ts_data.squeeze().name].values())
        model_forecasts = forecasts.predicted_mean.values.tolist()

        z_conf_int_95_upper = list(z_pred['outputs'][0]['conf_int_95_upper_'+ts_data.squeeze().name].values())
        model_conf_int_95_upper = forecasts.conf_int()['upper '+ts_data.squeeze().name].tolist()

        z_conf_int_95_lower = list(z_pred['outputs'][0]['conf_int_95_lower_'+ts_data.squeeze().name].values())
        model_conf_int_95_lower = forecasts.conf_int()['lower '+ts_data.squeeze().name].tolist()

        self.assertEqual(np.allclose(z_forecasts,model_forecasts),True)
        self.assertEqual(np.allclose(z_conf_int_95_upper, model_conf_int_95_upper),True)
        self.assertEqual(np.allclose(z_conf_int_95_lower, model_conf_int_95_lower),True)
        

    def test_4(self):
        data = self.getMultiDimensionalData()
        model = VARMAX(data,order=(1,2))
        result = model.fit()

        f_name='varmax_12.pmml'
        StatsmodelsToPmml(result, f_name,model_name="varmax_test",conf_int=[95])

        model_name = self.adapa_utility.upload_to_zserver(f_name)
        z_pred = self.adapa_utility.score_in_zserver(model_name, {'h':5},'TS')
        forecasts=result.get_forecast(5)

        z_forecast_hum = list(z_pred['outputs'][0]['predicted_SanDiegoHum'].values())
        model_forecast_hum = forecasts.predicted_mean['SanDiegoHum'].values.tolist()

        z_forecast_pressure = list(z_pred['outputs'][0]['predicted_SanDiegoPressure'].values())
        model_forecast_pressure = forecasts.predicted_mean['SanDiegoPressure'].values.tolist()

        z_forecast_temp = list(z_pred['outputs'][0]['predicted_SanDiegoTemp'].values())
        model_forecast_temp = forecasts.predicted_mean['SanDiegoTemp'].values.tolist()

        z_conf_int_95_lower_hum = list(z_pred['outputs'][0]['conf_int_95_lower_SanDiegoHum'].values())
        model_conf_int_95_lower_hum = forecasts.conf_int()['lower SanDiegoHum'].values.tolist()

        z_conf_int_95_lower_pressure = list(z_pred['outputs'][0]['conf_int_95_lower_SanDiegoPressure'].values())
        model_conf_int_95_lower_pressure = forecasts.conf_int()['lower SanDiegoPressure'].values.tolist()

        z_conf_int_95_lower_temp = list(z_pred['outputs'][0]['conf_int_95_lower_SanDiegoTemp'].values())
        model_conf_int_95_lower_temp = forecasts.conf_int()['lower SanDiegoTemp'].values.tolist()

        z_conf_int_95_upper_hum = list(z_pred['outputs'][0]['conf_int_95_upper_SanDiegoHum'].values())
        model_conf_int_95_upper_hum = forecasts.conf_int()['upper SanDiegoHum'].values.tolist()

        z_conf_int_95_upper_pressure = list(z_pred['outputs'][0]['conf_int_95_upper_SanDiegoPressure'].values())
        model_conf_int_95_upper_pressure = forecasts.conf_int()['upper SanDiegoPressure'].values.tolist()

        z_conf_int_95_upper_temp = list(z_pred['outputs'][0]['conf_int_95_upper_SanDiegoTemp'].values())
        model_conf_int_95_upper_temp = forecasts.conf_int()['upper SanDiegoTemp'].values.tolist()

        self.assertEqual(np.allclose(z_forecast_hum,model_forecast_hum),True)
        self.assertEqual(np.allclose(z_forecast_pressure,model_forecast_pressure),True)
        self.assertEqual(np.allclose(z_forecast_temp,model_forecast_temp),True)
        self.assertEqual(np.allclose(z_conf_int_95_lower_hum,model_conf_int_95_lower_hum),True)
        self.assertEqual(np.allclose(z_conf_int_95_lower_pressure,model_conf_int_95_lower_pressure),True)
        self.assertEqual(np.allclose(z_conf_int_95_lower_temp,model_conf_int_95_lower_temp),True)
        self.assertEqual(np.allclose(z_conf_int_95_upper_hum,model_conf_int_95_upper_hum),True)
        self.assertEqual(np.allclose(z_conf_int_95_upper_pressure,model_conf_int_95_upper_pressure),True)
        self.assertEqual(np.allclose(z_conf_int_95_upper_temp,model_conf_int_95_upper_temp),True)



    @classmethod
    def tearDownClass(cls):
        print("\n******* Finished *******\n")
     
if __name__ == '__main__':
    unittest.main(warnings='ignore')