from pprint import pprint
from nyoka.PMML43Ext import *
import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as smm
import statsmodels as sm
from statsmodels.tsa.statespace.kalman_filter import FilterResults
from statsmodels.tsa.statespace.kalman_smoother import SmootherResults
from statsmodels.tsa.tsatools import add_trend

def generate_statsmodels(pmml_file_name):

    def get_time_series_data_from_pmml(time_series_model_obj):
        time_series_obj = time_series_model_obj.get_TimeSeries()[0]
        time_values = time_series_obj.get_TimeValue()
        index = list()
        time_series_values = list()
        if(time_series_obj.get_usage() == "logical"):
            raise ValueError('Not implemented')
        else:
            if(time_values[0].hasContent_()):
                for time_value in time_values:
                    index.append(time_value.get_Timestamp().get_valueOf_())
                    time_series_values.append(time_value.get_value())
            else:
                for time_value in time_values:
                    index.append(time_value.get_index())
                    time_series_values.append(time_value.get_value())
                    
        time_series_data = pd.Series(data = time_series_values, index = index)
        return time_series_data

    def get_seasonal_model_from_pmml(ts_data, ts_model_obj):
        arima_obj = ts_model_obj.get_ARIMA()
        const = ts_model_obj.get_ARIMA().get_constantTerm()
        pred_method = ts_model_obj.get_ARIMA().get_predictionMethod()
        #Non Seasonal Component
        non_season_comp = ts_model_obj.get_ARIMA().get_NonseasonalComponent()
        p = non_season_comp.get_p()
        d = non_season_comp.get_d()
        q = non_season_comp.get_q()
        non_seasonal_ar_params = get_non_seasonal_ar_params(ts_model_obj)
        non_seasonal_ma_params = np.array([i.get_value() for i in non_season_comp.get_MA().get_Coefficients().get_Coefficient()] if q>0 else [])
        
        params = np.array([])
        if non_seasonal_ar_params.size:
            params=np.append(params,non_seasonal_ar_params)
        if non_seasonal_ma_params.size:
            params=np.append(params,non_seasonal_ma_params)
        #Seasonal Component
        season_comp = ts_model_obj.get_ARIMA().get_SeasonalComponent()
        P = season_comp.get_P()
        D = season_comp.get_D()
        Q = season_comp.get_Q()
        S = season_comp.get_period()
        seasonal_ar_params = get_seasonal_ar_params(ts_model_obj)
        seasonal_ma_params = np.array([i.get_value() for i in season_comp.get_MA().get_Coefficients().get_Coefficient()] if Q>0 else [])
        
        sigma2, cov_type, cov_kwds = get_seasonal_arima_extension_params(arima_obj)
        
        if seasonal_ar_params.size:
            params=np.append(params,seasonal_ar_params)
        if seasonal_ma_params.size:
            params=np.append(params,seasonal_ma_params)
            
        params = np.append(params,sigma2)
        
        model = smm.tsa.statespace.SARIMAX(endog=ts_data, order=(p, d, q), seasonal_order=(P,D,Q,S), enforce_stationarity=False,enforce_invertibility=False)

        model.polynomial_ar = get_structured_params(model.polynomial_ar, non_seasonal_ar_params, ar_or_ma = 'ar')
        model.polynomial_ma = get_structured_params(model.polynomial_ma, non_seasonal_ma_params, ar_or_ma = 'ma')
        model.polynomial_seasonal_ar = get_structured_params(model.polynomial_seasonal_ar, seasonal_ar_params, ar_or_ma = 'ar')
        model.polynomial_seasonal_ma = get_structured_params(model.polynomial_seasonal_ma, seasonal_ma_params, ar_or_ma = 'ma')
   
        result = model.smooth(params, transformed=True, cov_type = cov_type, cov_kwds=cov_kwds)
        return result, model

    def get_non_seasonal_model_from_pmml(ts_data, ts_model_obj):
        non_season_comp = ts_model_obj.get_ARIMA().get_NonseasonalComponent()
        const = ts_model_obj.get_ARIMA().get_constantTerm()
        pred_method = ts_model_obj.get_ARIMA().get_predictionMethod()
        p = non_season_comp.get_p()
        d = non_season_comp.get_d()
        q = non_season_comp.get_q()
        sigma2 = get_sigma(ts_model_obj)
        ar_params = get_non_seasonal_ar_params(ts_model_obj)
        ma_params = np.array([i.get_value() for i in non_season_comp.get_MA().get_Coefficients().get_Coefficient()] if q>0 else [])

        params = np.array([])
        params=np.append(params,const)
        if ar_params.size:
            params=np.append(params,ar_params)
        if ma_params.size:
            params=np.append(params,ma_params)

        methods = {"conditionalLeastSquares-exactLeastSquares": "css-mle",
                    "conditionalLeastSquares": "css",
                    "exactLeastSquares": "mle"}

        model = sm.tsa.arima_model.ARIMA(endog=ts_data, order=(p, d, q),exog=None, dates=None, freq=None)
        model.exog = make_exog(model.endog, model.exog, trend= 0 if const==0 else 1)
        model.transparams = False
        model.nobs = len(ts_data) if methods[pred_method]=="css" else len(ts_data)-d
        model.k_trend = 0 if const==0 else 1
        model.method = methods[pred_method]
        model.sigma2 = sigma2
            
        normalized_cov_params = None  # TODO: Update once updated in Statsmodels
        arima_fit = sm.tsa.arima_model.ARIMAResults(model, params, normalized_cov_params)
        arima_fit.k_diff = d
        return sm.tsa.arima_model.ARIMAResultsWrapper(arima_fit) , model

    def get_sigma(ts_model_obj):
        ExtensionList = ts_model_obj.get_ARIMA().get_Extension()
        if ExtensionList:  # if Extension elements exist in pmml file
            if ExtensionList[0].get_name() == 'sigmaSquare':
                sigma2 = np.float64(ExtensionList[0].get_value())
                return sigma2
        return 0

    def get_seasonal_arima_extension_params(arima_obj):
        sigma2 = 0
        cov_type = None
        cov_kwds = {}
        asBool = {'True': True, 'False': False}   
        ExtensionList = arima_obj.get_Extension()
        if ExtensionList:  # if Extension elements exist in pmml file
            for extension in ExtensionList:
                if extension.get_name() == 'sigmaSquare':
                    sigma2 = np.float64(extension.get_value())
                if extension.get_name() == 'cov_type':
                    cov_type = extension.get_value()
                if extension.get_name() == 'approx_complex_step':
                    cov_kwds['approx_complex_step'] = asBool[extension.get_value()]
                if extension.get_name() == 'approx_centered':
                    cov_kwds['approx_centered'] = asBool[extension.get_value()]
        return sigma2, cov_type, cov_kwds

    def get_non_seasonal_ar_params(ts_model_obj):
        non_season_comp = ts_model_obj.get_ARIMA().get_NonseasonalComponent()
        str_params = non_season_comp.get_AR().get_Array().get_valueOf_().strip()
        if len(str_params):
            ar_params = np.array(str_params.split(' '), dtype=np.float64)
        else:
            ar_params = np.array(list())
        return ar_params

    def get_seasonal_ar_params(ts_model_obj):
        season_comp = ts_model_obj.get_ARIMA().get_SeasonalComponent()
        str_params = season_comp.get_AR().get_Array().get_valueOf_().strip()
        if len(str_params):
            ar_params = np.array(str_params.split(' '), dtype=np.float64)
        else:
            ar_params = np.array(list())
        return ar_params

    def make_exog(endog, exog, trend):
        if exog is None and trend == 1:
            exog = np.ones((len(endog), 1))
        elif exog is not None and trend == 1:
            exog = add_trend(exog, trend='c', prepend=True)
        elif exog is not None and trend == 0:
            if exog.var() == 0:
                exog = None
        return exog
    
    def get_structured_params(x, y, ar_or_ma = None):
        out = np.array([x[0]])
        yindex = 0
        if(ar_or_ma == 'ar'):
            for i in range(1, len(x)):
                if(x[i] != 0.0):
                    out = np.append(out, x[i]*-y[yindex])
                    yindex += 1
                else:
                    out = np.append(out, x[i])
        elif(ar_or_ma == 'ma'):
            for i in range(1, len(x)):
                if(x[i] != 0.0):
                    out = np.append(out, x[i]*y[yindex])
                    yindex += 1
                else:
                    out = np.append(out, x[i])
        return out

    nyoka_pmml = parse(pmml_file_name, silence=True)
    nyoka_time_series_model_obj = nyoka_pmml.TimeSeriesModel[0]
    time_series_data = get_time_series_data_from_pmml(nyoka_time_series_model_obj)
    modelName = nyoka_time_series_model_obj.get_modelName()

    if(modelName == 'arima'):
        result, model = get_non_seasonal_model_from_pmml(time_series_data, nyoka_time_series_model_obj)
        return result, model

    elif(modelName == 'sarimax'):
        result, model = get_seasonal_model_from_pmml(time_series_data, nyoka_time_series_model_obj)
        return result, model

    else:
        raise ValueError('Model Not implemented')

