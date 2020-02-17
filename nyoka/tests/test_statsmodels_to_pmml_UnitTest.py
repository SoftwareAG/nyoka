
import sys,os

import pandas as pd
import numpy as np
from statsmodels.tsa.api import ARIMA, SARIMAX, ExponentialSmoothing, VARMAX
from statsmodels.tsa.arima.model import ARIMA as StateSpaceARIMA
import unittest

from nyoka import ArimaToPMML, ExponentialSmoothingToPMML


class TestMethods(unittest.TestCase):
    
    def getData1(self):
        # data with trend and seasonality present
        # no of international visitors in Australia
        data = [41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015, 31.1802, 37.7179,
                40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668, 35.1919,
                44.3197, 47.9137]
        index = pd.date_range(start='2005', end='2010-Q4', freq='QS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_visitors'
        ts_data = ts_data.to_frame()
        return ts_data
		
    def getData2(self):
		# data with trend but no seasonality
        # no. of annual passengers of air carriers registered in Australia
        data = [17.5534, 21.86, 23.8866, 26.9293, 26.8885, 28.8314, 30.0751, 30.9535, 30.1857, 31.5797, 32.5776,
                33.4774, 39.0216, 41.3864, 41.5966]
        index = pd.date_range(start='1990', end='2005', freq='A')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_passengers'
        return ts_data

    def getData3(self):
		# data with no trend and no seasonality
        # Oil production in Saudi Arabia
        data = [446.6565, 454.4733, 455.663, 423.6322, 456.2713, 440.5881, 425.3325, 485.1494, 506.0482, 526.792,
                514.2689, 494.211]
        index = pd.date_range(start='1996', end='2008', freq='A')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'oil_production'
        return ts_data

    def getData4(self):
		# Non Seasonal Data
        data = [266,146,183,119,180,169,232,225,193,123,337,186,194,150,210,273,191,287,
                226,304,290,422,265,342,340,440,316,439,401,390,490,408,490,420,520,480]
        index = pd.date_range(start='2016-01-01', end='2018-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'date_index'
        ts_data.name = 'cars_sold'
        ts_data = ts_data.to_frame()
        return ts_data

    def getData5(self):
		# Seasonal Data
        data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150,
                178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235,
                229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315,
                364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467,
                404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407,
                362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432]
        index = pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_passengers'
        ts_data = ts_data.to_frame()
        return ts_data

    def get_data_for_varmax(self):
        data = pd.read_csv("nyoka/tests/SanDiegoWeather.csv", parse_dates=True, index_col=0)
        return data
    
    #Exponential Smoothing Test cases
    
    def test_exponentialSmoothing_01(self):
        ts_data = self.getData1()        
        f_name='exponential_smoothing1.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=True, 
                                        seasonal='add', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(results_obj, f_name, model_name="Test2", description="test model")
        self.assertEqual(os.path.isfile(f_name),True)


    #Non Seasonal Arima Test cases
    def test_non_seasonal_arima1(self):
        ts_data = self.getData4()
        f_name='non_seasonal_arima1.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle')
        ArimaToPMML(result, f_name, model_name="arima_920")
        self.assertEqual(os.path.isfile(f_name),True)

    def test_non_seasonal_arima2(self):
        ts_data = self.getData4()
        f_name='non_seasonal_arima2.pmml'
        model = ARIMA(ts_data,order=(9, 2, 3))
        result = model.fit(trend = 'nc', method = 'css-mle')
        ArimaToPMML(result, f_name, description="A test model")
        self.assertEqual(os.path.isfile(f_name),True)

    def test_non_seasonal_arima3(self):
        ts_data = self.getData4()
        f_name='non_seasonal_arima3.pmml'
        model = ARIMA(ts_data,order=(1, 0, 1))
        result = model.fit(trend = 'c', method = 'css-mle')
        ArimaToPMML(result, f_name, description="A test model")
        self.assertEqual(os.path.isfile(f_name),True)

    def test_non_seasonal_arima4(self):
        ts_data = self.getData4()
        f_name='non_seasonal_arima4.pmml'
        model = StateSpaceARIMA(ts_data,order=(1, 0, 1),trend = 'c')
        result = model.fit()
        ArimaToPMML(result, f_name, description="A test model",conf_int=[80])
        self.assertEqual(os.path.isfile(f_name),True)

    def test_non_seasonal_arima5(self):
        ts_data = self.getData4()
        f_name='non_seasonal_arima5.pmml'
        model = StateSpaceARIMA(ts_data,order=(1, 1, 1))
        result = model.fit()
        ArimaToPMML(result, f_name, description="A test model")
        self.assertEqual(os.path.isfile(f_name),True)
    


    #Seasonal Arima Test cases
    def test_seasonal_arima1(self):
        ts_data = self.getData5()
        f_name='seasonal_arima1.pmml'
        model = SARIMAX(endog = ts_data,
                                        order = (1, 1, 1),
                                        seasonal_order = (1, 1, 1, 12),
                                        )
        result = model.fit(disp=False)
        ArimaToPMML(result, f_name)
        self.assertEqual(os.path.isfile(f_name),True)

    def test_seasonal_arima2(self):
        ts_data = self.getData5()
        f_name='seasonal_arima2.pmml'
        model = SARIMAX(endog = ts_data,
                                        order = (1, 0, 1),
                                        seasonal_order = (1, 1, 1, 12),
                                        )
        result = model.fit(disp=False)
        ArimaToPMML(result, f_name,conf_int=[95])
        self.assertEqual(os.path.isfile(f_name),True)

    def test_seasonal_arima3(self):
        ts_data = self.getData5()
        f_name='seasonal_arima3.pmml'
        model = SARIMAX(endog = ts_data,
                                        order = (1, 1, 1),
                                        seasonal_order = (1, 0, 1, 12),
                                        )
        result = model.fit(disp=False)
        ArimaToPMML(result, f_name)
        self.assertEqual(os.path.isfile(f_name),True)

    def test_seasonal_arima4(self):
        ts_data = self.getData5()
        f_name='seasonal_arima4.pmml'
        model = SARIMAX(endog = ts_data,
                                        order = (1, 0, 1),
                                        seasonal_order = (1, 0, 1, 12),
                                        )
        result = model.fit(disp=False)
        ArimaToPMML(result, f_name)
        self.assertEqual(os.path.isfile(f_name),True)

    def test_seasonal_arima5(self):
        ts_data = self.getData5()
        f_name='seasonal_arima5.pmml'
        model = SARIMAX(endog = ts_data,
                                        order = (0, 0, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 'c',
                                        )
        result = model.fit(disp=False)
        ArimaToPMML(result, f_name)
        self.assertEqual(os.path.isfile(f_name),True)

    def test_varmax_with_intercept(self):
        ts_data = self.get_data_for_varmax()
        f_name='varmax_with_intercept.pmml'
        model = VARMAX(ts_data, order=(1,1))
        result = model.fit()
        ArimaToPMML(result, f_name, conf_int=[80,95])
        self.assertEqual(os.path.isfile(f_name),True)

    def test_varmax_without_intercept(self):
        ts_data = self.get_data_for_varmax()
        f_name='varmax_without_intercept.pmml'
        model = VARMAX(ts_data, order=(1,1), trend=None)
        result = model.fit()
        ArimaToPMML(result, f_name)
        self.assertEqual(os.path.isfile(f_name),True)


if __name__=='__main__':
    unittest.main(warnings='ignore')
