from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from pprint import pprint
from PMML43Ext import *
import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as sm


class ArimaToPMML:
    """

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

        def get_arima_obj(sm_model, sm_results):
            """
            return an instance of nyoka's  ARIMA() from statsmodels(sm) arima model object
            """
            p = sm_results._results.k_ar
            if 'k_diff' in sm_results._results.__dict__.keys():
                d = sm_results._results.k_diff
            else:
                d = 0
            q = sm_results._results.k_ma
            if sm_model.method == 'css':
                pred_method = "conditionalLeastSquares"
            elif sm_model.method == 'mle':
                pred_method = "exactLeastSquares"
            else:
                pred_method = None

            ar_params = ArrayType(n=p)
            content_value = ' '.join([str(i) for i in sm_results._results.arparams])
            ar_params.content_[0].value = content_value

            ma_coeffs = [Coefficient(value=coeff) for coeff in sm_results._results.maparams]
            ny_ma_obj = MA(Coefficients=Coefficients(numberOfCoefficients=q, Coefficient=ma_coeffs))

            ny_arima_obj = ARIMA(
                constantTerm=sm_results.params['const'],
                predictionMethod=pred_method,
                NonseasonalComponent=NonseasonalComponent(
                    p=p, d=d, q=q, AR=AR(Array=ar_params), MA=ny_ma_obj
                )
            )
            return ny_arima_obj

        n_columns = 1  # because we are dealing with Series object
        n_samples = time_series_data.size
        function_name = 'timeSeries'
        best_fit = 'ARIMA'

        pmml = PMML(
            version='4.4',
            Header=Header(
                copyright="Copyright (c) 2017 PB&RB", description="ARIMA Model",
                Application=Application(name="Rattle/PMML", version="1.3"), Timestamp=Timestamp(datetime.utcnow())
            ),
            DataDictionary=DataDictionary(numberOfFields=n_columns, DataField=get_data_field_objs(time_series_data)),
            TimeSeriesModel=[TimeSeriesModel(
                modelName='arima',
                functionName=function_name, bestFit=best_fit, isScorable=True,
                MiningSchema=MiningSchema(MiningField=get_mining_field_objs(time_series_data)),
                TimeSeries=[TimeSeries(
                    usage='original', startTime=0, endTime=n_samples - 1, interpolationMethod='none',
                    TimeValue=get_time_value_objs(time_series_data)
                )],
                ARIMA=get_arima_obj(model_obj, results_obj)
            )]
        )

        # print(len(get_data_field_objs(time_series_data)))
        pmml.export(outfile=open(pmml_file_name, "w"), level=0)


def reconstruct(pmml_f_name):

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

    def get_ar_params(ts_model_obj):
        non_season_comp = ts_model_obj.get_ARIMA().get_NonseasonalComponent()
        str_params = non_season_comp.get_AR().get_Array().get_valueOf_().strip()
        if len(str_params):
            ar_params = np.array(str_params.split(' '), dtype=np.float64)
        else:
            ar_params = np.array(list())
        return ar_params

    def get_params_model_from_pmml(ts_data, ts_model_obj):
        non_season_comp = ts_model_obj.get_ARIMA().get_NonseasonalComponent()
        const = ts_model_obj.get_ARIMA().get_constantTerm()
        p = non_season_comp.get_p()
        d = non_season_comp.get_d()
        q = non_season_comp.get_q()
        ar_params = get_ar_params(ts_model_obj)
        ma_params = np.array([i.get_value() for i in non_season_comp.get_MA().get_Coefficients().get_Coefficient()])
        stsmdl_arima = sm.tsa.ARIMA(endog=ts_data, order=(p, d, q))
        results = stsmdl_arima.fit()
        params = list()
        params.append(const)
        if not ar_params.size:
            params.extend(ar_params)
        if not ma_params.size:
            params.extend(ma_params)
        return results, stsmdl_arima

    nyoka_pmml = parse(pmml_f_name, silence=False)
    ny_ts_model_obj = nyoka_pmml.TimeSeriesModel[0]
    ts_data = get_ts_data_from_pmml(ny_ts_model_obj)
    results, stsmdl = get_params_model_from_pmml(ts_data, ny_ts_model_obj)
    return results, stsmdl



