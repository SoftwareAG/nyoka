"""
 Copyright (c) 2004-2016 Zementis, Inc.
 Copyright (c) 2016-2021 Software AG, Darmstadt, Germany and/or Software AG USA Inc., Reston, VA, USA, and/or its

 SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 """
from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from PMML44 import *
from datetime import datetime
import metadata
import warnings
from base.constants import *

class ExponentialSmoothingToPMML:
    """
    Write a PMML file using model-object, model-parameters and time series data. Models are built using Statsmodels.

    Parameters:
    -----------
    results_obj: 
        Instance of HoltWintersResults from statsmodels
    pmml_file_name: string
        Name of the PMML
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description of the model
    """
    def __init__(self, results_obj=None, pmml_file_name="from_ExponentialSmoothing.pmml", model_name=None, description=None):

        def get_time_value_objs():
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
            ts_int_index = range(len(results_obj.model.endog))
            time_value_objs = list()
            for int_idx in ts_int_index:
                time_value_objs.append(TimeValue(index=int_idx, value=results_obj.model.endog[int_idx]))
            return time_value_objs

        def get_pmml_datatype_optype(series_obj):
            pmml_data_type = None
            pmml_op_type = 'continuous'
            if str(series_obj.dtype) in ['datetime64[ns]', 'datetime64[ns, tz]', 'timedelta[ns]']:
                pmml_data_type = DATATYPE.DATETIME
            elif str(series_obj.dtype) == 'float32':
                pmml_data_type = DATATYPE.FLOAT
            elif str(series_obj.dtype) == 'float64':
                pmml_data_type = DATATYPE.DOUBLE
            elif str(series_obj.dtype) in ['int64', 'int32']:
                pmml_data_type = DATATYPE.INTEGER

            return pmml_data_type, pmml_op_type

        def get_data_field_objs():
            """
            Create a list with instances of DataField
            """
            data_field_objs = list()
            index_name = results_obj.data.orig_endog.index.name
            idx_data_type, idx_op_type = get_pmml_datatype_optype(results_obj.model._index)
            data_field_objs.append(DataField(name=index_name, dataType=idx_data_type, optype=idx_op_type))
            if results_obj.data.orig_endog.__class__.__name__ == 'DataFrame':
                ts_name = results_obj.data.orig_endog.columns[0]
            elif results_obj.data.orig_endog.__class__.__name__ == 'Series':
                ts_name = results_obj.data.orig_endog.name
            else:
                ts_name = 'input'
            ts_data_type, ts_op_type = get_pmml_datatype_optype(results_obj.model.endog)
            data_field_objs.append(DataField(name=ts_name, dataType=ts_data_type, optype=ts_op_type))
            return data_field_objs

        def get_mining_field_objs():
            """
            Create a list with instances of MiningField
            """
            mining_field_objs = list()
            if results_obj.data.orig_endog.__class__.__name__ == 'DataFrame':
                ts_name = results_obj.data.orig_endog.columns[0]
            elif results_obj.data.orig_endog.__class__.__name__ == 'Series':
                ts_name = results_obj.data.orig_endog.name
            else:
                ts_name = 'input'
            idx_name = results_obj.data.orig_endog.index.name
            idx_usage_type = FIELD_USAGE_TYPE.ORDER
            mining_field_objs.append(MiningField(name=idx_name, usageType=idx_usage_type))
            ts_usage_type = FIELD_USAGE_TYPE.TARGET
            mining_field_objs.append(MiningField(name=ts_name, usageType=ts_usage_type))
            return mining_field_objs

        n_samples = results_obj.model.nobs
        n_columns = 1  # because we are dealing with Series object
        function_name = MINING_FUNCTION.TIMESERIES
        best_fit = TIMESERIES_ALGORITHM.EXPONENTIAL_SMOOTHING
        extension_objs = list()
        alpha = results_obj.params['smoothing_level']  # alpha is smoothing parameter for level
        level_smooth_val = results_obj.level[-1]  # smoothed level at last time-index
        initial_level = results_obj.params['initial_level']
        # extension_objs.append(Extension(name='initialLevel', value=initial_level))
        import numpy as np
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
        if results_obj.model.trend:  # model_obj.trend can take values in {'add', 'mul', None}
            trend_smooth_val = results_obj.slope[-1]
            initial_trend = results_obj.params['initial_slope']
            if results_obj.model.trend == 'add':
                if results_obj.model.damped:
                    trend_type = EXPONENTIAL_SMOOTHING_TREND.DAMPED_ADDITIVE
                else:
                    trend_type = EXPONENTIAL_SMOOTHING_TREND.ADDITIVE
            else:  # model_obj.trend == 'mul':
                if results_obj.model.damped:
                    trend_type = EXPONENTIAL_SMOOTHING_TREND.DAMPED_MULTIPLICATIVE
                else:
                    trend_type = EXPONENTIAL_SMOOTHING_TREND.MULTIPLICATIVE
            trend_obj = Trend_ExpoSmooth(trend=trend_type, gamma=gamma, phi=phi, smoothedValue=trend_smooth_val)
            # extension_objs.append(Extension(name='initialTrend', value=initial_trend))
        else:
            trend_obj = None
        if results_obj.model.seasonal:  # model_obj.seasonal can take values in {'add', 'mul', None}
            period = results_obj.model.seasonal_periods
            initial_seasons = ArrayType(n=period, type_ = ARRAY_TYPE.REAL)
            content_value = ' '.join([str(i) for i in results_obj.params['initial_seasons']])
            initial_seasons.content_[0].value = content_value
            if results_obj.model.seasonal == 'add':
                seasonal_type = EXPONENTIAL_SMOOTHING_SEASONALITY.ADDITIVE
            else:  # model_obj.seasonal == 'mul':
                seasonal_type = EXPONENTIAL_SMOOTHING_SEASONALITY.MULTIPLICATIVE
            season_obj = Seasonality_ExpoSmooth(type_=seasonal_type, period=period, delta=delta,
                                                Array=initial_seasons)
        else:
            season_obj = None
        pmml = PMML(
            version=PMML_SCHEMA.VERSION,
            Header=Header(
                copyright=HEADER_INFO.COPYRIGHT,
                description=description if description else HEADER_INFO.DEFAULT_DESCRIPTION,
                Timestamp=Timestamp(datetime.now()),
                Application=Application(name=HEADER_INFO.APPLICATION_NAME,version=HEADER_INFO.APPLICATION_VERSION)
            ),
            DataDictionary=DataDictionary(numberOfFields=n_columns, DataField=get_data_field_objs()),
            TimeSeriesModel=[TimeSeriesModel(
                modelName= model_name if model_name else 'simple exponential smoothing',
                functionName=function_name, bestFit=best_fit, isScorable=True,
                MiningSchema=MiningSchema(MiningField=get_mining_field_objs()),
                TimeSeries=[TimeSeries(
                    usage=TIMESERIES_USAGE.ORIGINAL, startTime=0, endTime=n_samples - 1, interpolationMethod='none',
                    TimeValue=get_time_value_objs()
                )],
                ExponentialSmoothing=ExponentialSmoothing(
                    Level=Level(alpha=alpha, smoothedValue=level_smooth_val),
                    Trend_ExpoSmooth=trend_obj,
                    Seasonality_ExpoSmooth=season_obj,
                ),
                # Extension=extension_objs
            )]
        )
        pmml.export(outfile=open(pmml_file_name, "w"), level=0)
