from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from nyoka.statsmodels import arima
from nyoka.statsmodels import exponential_smoothing

def statsmodels_to_pmml(time_series_data, model_obj, results_obj, pmml_file_name):
    if model_obj.__class__.__name__ == 'ExponentialSmoothing':
        exponential_smoothing.ExponentialSmoothingToPMML(time_series_data, model_obj, results_obj, pmml_file_name)
    elif model_obj.__class__.__name__ in ['ARIMA','SARIMAX']:
        arima.ArimaToPMML(time_series_data, model_obj, results_obj, pmml_file_name)
    else:
        raise NotImplementedError("Currently supports ExponentialSmoothing, ARIMA, SARIMAX only.")