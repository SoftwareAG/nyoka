import sys, os
import pandas as pd
import numpy as np
from statsmodels.tsa.base import tsa_model as tsa
from statsmodels.tsa import holtwinters as hw
import unittest
from nyoka import ExponentialSmoothingToPMML 


class TestMethods(unittest.TestCase):

    
    def test_keras_01(self):

        def import_data(trend=False, seasonality=False):
            """
            Returns a dataframe with time series values.
            :param trend: boolean
                If True, returns data with trend
            :param seasonality: boolean
                If True, returns data with seasonality
            :return: ts_data: DataFrame
                Index of the data frame is either a time-index or an integer index. First column has time series values
            """
            if trend and seasonality:
                # no of international visitors in Australia
                data = [41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015, 31.1802, 37.7179,
                        40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668, 35.1919,
                        44.3197, 47.9137]
                index = pd.DatetimeIndex(start='2005', end='2010-Q4', freq='QS')
                ts_data = pd.Series(data, index)
                ts_data.index.name = 'datetime_index'
                ts_data.name = 'n_visitors'
                return ts_data
            elif trend:
                # no. of annual passengers of air carriers registered in Australia
                data = [17.5534, 21.86, 23.8866, 26.9293, 26.8885, 28.8314, 30.0751, 30.9535, 30.1857, 31.5797, 32.5776,
                        33.4774, 39.0216, 41.3864, 41.5966]
                index = pd.DatetimeIndex(start='1990', end='2005', freq='A')
                ts_data = pd.Series(data, index)
                ts_data.index.name = 'datetime_index'
                ts_data.name = 'n_passengers'
                return ts_data
            elif seasonality:
                pass
            else:
                # Oil production in Saudi Arabia
                data = [446.6565, 454.4733, 455.663, 423.6322, 456.2713, 440.5881, 425.3325, 485.1494, 506.0482, 526.792,
                        514.2689, 494.211]
                index = pd.DatetimeIndex(start='1996', end='2008', freq='A')
                ts_data = pd.Series(data, index)
                ts_data.index.name = 'datetime_index'
                ts_data.name = 'oil_production'
                return ts_data

        ts_data = import_data(trend=False, seasonality=False)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend=None, 
                                    damped=False, 
                                    seasonal=None, 
                                    seasonal_periods=None)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)

        ts_data = import_data(trend=True, seasonality=False)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend='add', 
                                    damped=False, 
                                    seasonal=None, 
                                    seasonal_periods=None)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend='mul', 
                                    damped=False, 
                                    seasonal=None, 
                                    seasonal_periods=None)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend='mul', 
                                    damped=True, 
                                    seasonal=None, 
                                    seasonal_periods=None)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)

        ts_data = import_data(trend=True, seasonality=True)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend='add', 
                                    damped=False, 
                                    seasonal='add', 
                                    seasonal_periods=4)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend='add', 
                                    damped=True, 
                                    seasonal='add', 
                                    seasonal_periods=4)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend='add', 
                                    damped=False, 
                                    seasonal='mul', 
                                    seasonal_periods=4)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)

        model_obj = hw.ExponentialSmoothing(ts_data, 
                                    trend='add', 
                                    damped=False, 
                                    seasonal='mul', 
                                    seasonal_periods=4)

        results_obj = model_obj.fit(optimized=True)

        ExponentialSmoothingToPMML(ts_data, model_obj, results_obj, 'exponential_smoothing.pmml')

        self.assertEqual(os.path.isfile("exponential_smoothing.pmml"),True)


if __name__=='__main__':
    unittest.main(warnings='ignore')







