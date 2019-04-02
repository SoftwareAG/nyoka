from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from PMML43Ext import *
import numpy as np
from datetime import datetime


class ExponentialSmoothingToPMML:
    """
    Write a PMML file using model-object, model-parameters and time series data. Models are built using Statsmodels.

    Parameters:
    -----------
    time_series_data: 
        Pandas Series object
    model_obj: 
        Instance of ExponentialSmoothing from statsmodels
    results_obj: 
        Instance of HoltWintersResults from statsmodels
    pmml_file_name: string
    """
    def __init__(self, time_series_data, model_obj, results_obj, pmml_file_name):

        def get_time_value_objs(ts_data):
            """
            Does not have time attribute

            Parameters:
            -----------
            ts_data: pandas Series

            Returns:
            --------
            time_value_objs: list
                Instances of TimeValue
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
            Create a list with instances of DataField
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
            Create a list with instances of MiningField
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
            initial_seasons = ArrayType(n=period, type_ = 'real')
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
                Timestamp=Timestamp(datetime.utcnow())
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
