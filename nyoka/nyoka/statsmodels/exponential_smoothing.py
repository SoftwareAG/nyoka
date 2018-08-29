from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from PMML43Ext import *
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa import holtwinters as hw


class ExponentialSmoothingToPMML:
    """
    Write a PMML file using model-object, model-parameters and time series data. Models are built using Statsmodels.
    :param time_series_data: Pandas Series object
    :param model_obj: Instance of ExponentialSmoothing() from statsmodels
    :param results_obj: Instance of HoltWintersResults() from statsmodels
    :param pmml_file_name:
    """
    def __init__(self, time_series_data, model_obj, results_obj, pmml_file_name):

        def get_time_value_objs(ts_data):
            """
            Does not have time attribute
            :param ts_data: pandas Series
            :return: time_value_objs: list
                Instances of TimeValue()
            """
            ts_int_index = range(len(ts_data))
            time_value_objs = list()
            for int_idx in ts_int_index:
                time_value_objs.append(TimeValue(index=int_idx, value=ts_data.iat[int_idx]))
            return time_value_objs

        def get_pmml_datatype_optype(series_obj):
            pmml_data_type = None
            pmml_op_type = None
            if str(series_obj.dtype) in {'datetime64[ns]', 'datetime64[ns, tz]', 'timedelta[ns]'}:
                pmml_data_type = 'dateTime'
                pmml_op_type = 'continuous'
            elif str(series_obj.dtype) == 'float32':
                pmml_data_type = 'float'
                pmml_op_type = 'continuous'
            elif str(series_obj.dtype) == 'float64':
                pmml_data_type = 'double'
                pmml_op_type = 'continuous'
            elif str(series_obj.dtype) in {'int64', 'int32'}:
                pmml_data_type = 'integer'
                pmml_op_type = 'continuous'
            return pmml_data_type, pmml_op_type

        def get_data_field_objs(ts_data):
            """
            Create a list with instances of DataField()
            """
            data_field_objs = list()
            index_name = ts_data.index.name
            idx_data_type, idx_op_type = get_pmml_datatype_optype(ts_data.index)
            data_field_objs.append(DataField(name=index_name, dataType=idx_data_type, optype=idx_op_type))
            ts_name = ts_data.name
            ts_data_type, ts_op_type = get_pmml_datatype_optype(ts_data)
            data_field_objs.append(DataField(name=ts_name, dataType=ts_data_type, optype=ts_op_type))
            return data_field_objs

        def get_mining_field_objs(ts_data):
            """
            Create a list with instances of MiningField()
            """
            mining_field_objs = list()
            idx_name = ts_data.index.name
            idx_usage_type = 'order'
            mining_field_objs.append(MiningField(name=idx_name, usageType=idx_usage_type))
            ts_name = ts_data.name
            ts_usage_type = 'target'
            mining_field_objs.append(MiningField(name=ts_name, usageType=ts_usage_type))
            return mining_field_objs

        n_samples = time_series_data.size
        n_columns = 1  # because we are dealing with Series object
        function_name = 'timeSeries'
        best_fit = 'ExponentialSmoothing'
        extension_objs = list()
        alpha = results_obj.params['smoothing_level']  # alpha is smoothing parameter for level
        level_smooth_val = results_obj.level[-1]  # smoothed level at last time-index
        initial_level = results_obj.params['initial_level']
        # extension_objs.append(Extension(name='initialLevel', value=initial_level))
        if np.isnan(results_obj.params['smoothing_slope']):
            gamma = None
        else:
            gamma = results_obj.params['smoothing_slope']  # gamma is smoothing parameter for trend
        if np.isnan(results_obj.params['smoothing_seasonal']):
            delta = None
        else:
            delta = results_obj.params['smoothing_seasonal']  # delta is smoothing parameter for seasonality
        if np.isnan(results_obj.params['damping_slope']):
            phi = 1
        else:
            phi = results_obj.params['damping_slope']  # damping parameter; which is applied on trend/slope
        if model_obj.trend:  # model_obj.trend can take values in {'add', 'mul', None}
            trend_smooth_val = results_obj.slope[-1]
            initial_trend = results_obj.params['initial_slope']
            if model_obj.trend == 'add':
                if model_obj.damped:
                    trend_type = 'damped_additive'
                else:
                    trend_type = 'additive'
            else:  # model_obj.trend == 'mul':
                if model_obj.damped:
                    trend_type = 'damped_multiplicative'
                else:
                    trend_type = 'multiplicative'
            trend_obj = Trend_ExpoSmooth(trend=trend_type, gamma=gamma, initialTrendValue=initial_trend, phi=phi,
                                         smoothedValue=trend_smooth_val)
            # extension_objs.append(Extension(name='initialTrend', value=initial_trend))
        else:
            trend_obj = None
        if model_obj.seasonal:  # model_obj.seasonal can take values in {'add', 'mul', None}
            period = model_obj.seasonal_periods
            initial_seasons = ArrayType(n=period)
            content_value = ' '.join([str(i) for i in results_obj.params['initial_seasons']])
            initial_seasons.content_[0].value = content_value
            if model_obj.seasonal == 'add':
                seasonal_type = 'additive'
            else:  # model_obj.seasonal == 'mul':
                seasonal_type = 'multiplicative'
            season_obj = Seasonality_ExpoSmooth(type_=seasonal_type, period=period, delta=delta,
                                                Array=initial_seasons)
        else:
            season_obj = None
        pmml = PMML(
            version='4.4',
            Header=Header(
                copyright="Copyright (c) 2017 PB&RB", description="Exponential Smoothing Model",
                # Extension=[Extension(name="user", value="tom", extender="Rattle/PMML")],
                Application=Application(name="Rattle/PMML", version="1.3"), Timestamp=Timestamp(datetime.utcnow())
            ),
            DataDictionary=DataDictionary(numberOfFields=n_columns, DataField=get_data_field_objs(time_series_data)),
            TimeSeriesModel=[TimeSeriesModel(
                modelName='simple exponential smoothing',
                functionName=function_name, bestFit=best_fit, isScorable=True,
                MiningSchema=MiningSchema(MiningField=get_mining_field_objs(time_series_data)),
                TimeSeries=[TimeSeries(
                    usage='original', startTime=0, endTime=n_samples - 1, interpolationMethod='none',
                    TimeValue=get_time_value_objs(time_series_data)
                )],
                ExponentialSmoothing=ExponentialSmoothing(
                    Level=Level(alpha=alpha, initialLevelValue=initial_level, smoothedValue=level_smooth_val),
                    Trend_ExpoSmooth=trend_obj,
                    Seasonality_ExpoSmooth=season_obj,
                ),
                # Extension=extension_objs
            )]
        )
        pmml.export(outfile=open(pmml_file_name, "w"), level=0)


def reconstruct_expon_smooth(pmml_file_name):
    """
    Parses a pmml file and extracts the parameters and time series data. Uses Statsmodels to instantiate a model object
    of exponential smoothing. Returns parameters and model-object
    :param pmml_file_name:
    :return: params: dictionary
        Parameters of the model as key-value pairs
    :return: stsmdl: model object from Statsmodels
        This model object is created using the parameters and time-series data extracted from the pmml file
    """

    def get_ts_data_from_pmml(ts_model_obj):
        time_series_obj = ts_model_obj.get_TimeSeries()[0]
        time_values = time_series_obj.get_TimeValue()
        index = list()
        ts_values = list()
        for time_value in time_values:
            index.append(time_value.get_index())
            ts_values.append(time_value.get_value())
        ts_data = pd.Series(data=ts_values, index=index)
        return ts_data

    def get_params_model_from_pmml(ts_data, ts_model_obj):
        params = dict()
        exp_smooth_obj = ts_model_obj.get_ExponentialSmoothing()
        level_obj = exp_smooth_obj.get_Level()
        trend_obj = exp_smooth_obj.get_Trend_ExpoSmooth()
        season_obj = exp_smooth_obj.get_Seasonality_ExpoSmooth()
        params['smoothing_level'] = level_obj.get_alpha()
        params['initial_level'] = level_obj.get_initialLevelValue()
        if trend_obj:
            params['smoothing_slope'] = trend_obj.get_gamma()
            params['damping_slope'] = trend_obj.get_phi()
            params['initial_slope'] = trend_obj.get_initialTrendValue()
            trend_type = trend_obj.get_trend()
            if trend_type == 'additive':
                trend = 'add'
                damped = False
            elif trend_type == 'multiplicative':
                trend = 'mul'
                damped = False
            elif trend_type == 'damped_additive':
                trend = 'add'
                damped = True
            elif trend_type == 'damped_multiplicative':
                trend = 'mul'
                damped = True
            elif trend_type == 'polynomial_exponential':
                pass
        else:
            trend = None
            damped = False
        if season_obj:
            params['smoothing_seasonal'] = season_obj.get_delta()
            seasonal_periods = season_obj.get_Array().get_n()
            params['initial_seasons'] = np.array(season_obj.get_Array().get_valueOf_().strip().split(' '))
            season_type = season_obj.get_type()
            if season_type == 'additive':
                seasonal = 'add'
            elif season_type == 'multiplicative':
                seasonal = 'mul'
        else:
            seasonal = None
            seasonal_periods = None
        # if ts_model_obj.get_Extension():  # if Extension elements exist in pmml file
        #     for extension in ts_model_obj.get_Extension():
        #        if extension.get_name() == 'initialLevel':
        #            params['initial_level'] = extension.get_value()
        #        if extension.get_name() == 'initialTrend':
        #            params['initial_slope'] = extension.get_value()
        stsmdl = hw.ExponentialSmoothing(ts_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
                                         damped=damped)
        return params, stsmdl

    nyoka_pmml = parse(pmml_file_name, silence=False)
    ts_model_obj = nyoka_pmml.TimeSeriesModel[0]
    ts_data = get_ts_data_from_pmml(ts_model_obj)
    params, stsmdl = get_params_model_from_pmml(ts_data, ts_model_obj)
    return params, stsmdl
