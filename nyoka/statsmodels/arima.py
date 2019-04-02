from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from pprint import pprint
from PMML43Ext import *
from datetime import datetime

class ArimaToPMML:
    def __init__(self, time_series_data, model_obj, results_obj, pmml_file_name):

        def ExportToPMML(model_name = None, arima_obj = None):
            n_columns = 1  # because we are dealing with Series object
            n_samples = time_series_data.size
            function_name = 'timeSeries'
            best_fit = 'ARIMA'

            pmml = PMML(
                version = '4.4',
                Header = Header(copyright = "Copyright (c) 2017 PB&RB", description = "ARIMA Model",  
                                Timestamp = Timestamp(datetime.utcnow())),
                DataDictionary = DataDictionary(numberOfFields = n_columns, 
                                                DataField = get_data_field_objs(time_series_data)),
                TimeSeriesModel = [ TimeSeriesModel(modelName = model_name,
                                                    functionName = function_name, bestFit = best_fit, isScorable = True,
                                                    MiningSchema = MiningSchema(MiningField = get_mining_field_objs(time_series_data)),
                                                    TimeSeries = get_time_series_obj_list(time_series_data, usage = 'original' , timeRequired = True),
                                                    ARIMA = arima_obj) ])

            pmml.export(outfile=open(pmml_file_name, "w"), level=0)

        def get_sarimax_obj(sm_model, sm_results):
            #NonSeasonal
            p = sm_results._results.specification.k_ar
            if 'k_diff' in sm_results._results.specification.__dict__.keys():
                d = sm_results._results.specification.k_diff
            else:
                d = 0
            q = sm_results._results.specification.k_ma
            ar_params = ArrayType(n = p, type_ = 'real')
            content_value = ' '.join([str(i) for i in sm_results._results._params_ar] if int(p) > 0 else [])
            ar_params.content_[0].value = content_value
            ma_coeffs = [Coefficient(value=coeff) for coeff in sm_results._results._params_ma] if int(q) > 0 else []
            ny_ma_obj = MA(Coefficients=Coefficients(numberOfCoefficients=q, Coefficient=ma_coeffs)) if q > 0 else None

            #Seasonal
            P = sm_results._results.specification.seasonal_order[0]
            D = sm_results._results.specification.seasonal_order[1]
            Q = sm_results._results.specification.seasonal_order[2]
            S = sm_results._results.specification.seasonal_periods
            seasonal_ar_params = ArrayType(n = P, type_ = 'real')
            seasonal_content_value = ' '.join([str(i) for i in sm_results._results._params_seasonal_ar] if int(P) > 0 else [])
            seasonal_ar_params.content_[0].value = seasonal_content_value
            seasonal_ma_coeffs = [Coefficient(value=coeff) for coeff in sm_results._results._params_seasonal_ma] if int(Q) > 0 else []
            ny_seasonal_ma_obj = MA(Coefficients=Coefficients(numberOfCoefficients=Q, Coefficient=seasonal_ma_coeffs)) if Q > 0 else None

            nyoka_sarimax_obj = ARIMA(#predictionMethod = None,
                                Extension = get_sarimax_extension_list(sm_results),
                                NonseasonalComponent = NonseasonalComponent(p = p, d = d, q = q, AR = AR(Array = ar_params), MA = ny_ma_obj),
                                SeasonalComponent = SeasonalComponent(P = P, D = D, Q = Q, period = S, AR = AR(Array = seasonal_ar_params), MA = ny_seasonal_ma_obj))
            return nyoka_sarimax_obj

        def get_arima_obj(sm_model, sm_results):
            p = sm_results._results.k_ar
            if 'k_diff' in sm_results._results.__dict__.keys():
                d = sm_results._results.k_diff
            else:
                d = 0
            q = sm_results._results.k_ma
            if sm_model.method == 'css-mle':
                pred_method = "conditionalLeastSquares-exactLeastSquares"
            elif sm_model.method == 'css':
                pred_method = "conditionalLeastSquares"
            elif sm_model.method == 'mle':
                pred_method = "exactLeastSquares"
            else:
                pred_method = None

            ar_params = ArrayType(n = p, type_ = 'real')
            content_value = ' '.join([str(i) for i in sm_results._results.arparams])
            ar_params.content_[0].value = content_value
            ma_coeffs = [Coefficient(value = coeff) for coeff in sm_results._results.maparams]
            ny_ma_obj = MA(Coefficients = Coefficients(numberOfCoefficients = q, Coefficient = ma_coeffs)) if q > 0 else None

            nyoka_arima_obj = ARIMA(constantTerm = sm_results.params[0],
                                predictionMethod = pred_method,
                                Extension = get_arima_extension_list(sm_model),
                                NonseasonalComponent = NonseasonalComponent(p = p, d = d, q = q, AR = AR(Array = ar_params), MA = ny_ma_obj))
            return nyoka_arima_obj

        def get_time_series_obj_list(ts_data, usage = 'original' , timeRequired = True):
            get_time_series_obj_list = list()
            def get_time_value_objs(time_series_data):
                ts_int_index = range(len(ts_data))
                time_value_objs = list()
                if(usage == 'original' and timeRequired == True):
                     for int_idx in ts_int_index:
                        time_value_objs.append(TimeValue(index = int_idx, value = ts_data.iat[int_idx], Timestamp = Timestamp(ts_data.index[int_idx])))
                elif(usage == 'logical' and timeRequired == True):
                    #TODO: Implement This
                    raise NotImplementedError("Not Implemented")
                return time_value_objs

            obj = TimeSeries(usage = usage, startTime = 0, endTime = time_series_data.size - 1, interpolationMethod = 'none', TimeValue = get_time_value_objs(time_series_data))
            get_time_series_obj_list.append(obj)
            return get_time_series_obj_list

        def get_mining_field_objs(ts_data):
            mining_field_objs = list()
            idx_name = ts_data.index.name
            idx_usage_type = 'order'
            mining_field_objs.append(MiningField(name = idx_name, usageType = idx_usage_type))
            ts_name = ts_data.name
            ts_usage_type = 'target'
            mining_field_objs.append(MiningField(name = ts_name, usageType = ts_usage_type))
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
            data_field_objs = list()
            index_name = ts_data.index.name
            idx_data_type, idx_op_type = get_pmml_datatype_optype(ts_data.index)
            data_field_objs.append(DataField(name=index_name, dataType=idx_data_type, optype=idx_op_type))
            ts_name = ts_data.name
            ts_data_type, ts_op_type = get_pmml_datatype_optype(ts_data)
            data_field_objs.append(DataField(name=ts_name, dataType=ts_data_type, optype=ts_op_type))
            return data_field_objs

        def get_sarimax_extension_list(results):
            extensions = list()
            extensions.append(Extension(name="sigmaSquare", value = results._params_variance[0], anytypeobjs_ = ['']))
            extensions.append(Extension(name="cov_type", value = results.cov_type, anytypeobjs_ = ['']))
            extensions.append(Extension(name="approx_complex_step", value = results._cov_approx_complex_step, anytypeobjs_ = ['']))
            extensions.append(Extension(name="approx_centered", value = results._cov_approx_centered, anytypeobjs_ = ['']))
            return extensions

        def get_arima_extension_list(model):
            extensions = list()
            extensions.append(Extension(name="sigmaSquare", value = model.sigma2, anytypeobjs_ = ['']))
            return extensions
        
        if(time_series_data.__class__.__name__ == 'DataFrame'):
            time_series_data = time_series_data.T.squeeze()

        if(model_obj.__class__.__name__ == 'SARIMAX' and results_obj.__class__.__name__ == 'SARIMAXResultsWrapper'):
            #Get SArimaX Object and Export
            sarimax_obj = get_sarimax_obj(model_obj, results_obj)
            model_name = 'sarimax'
            ExportToPMML(model_name = model_name, arima_obj = sarimax_obj)

        elif(model_obj.__class__.__name__ == 'ARIMA' and results_obj.__class__.__name__ == 'ARIMAResultsWrapper'):
            #Get Arima Object and Export
            arima_obj = get_arima_obj(model_obj, results_obj)
            model_name = 'arima'
            ExportToPMML(model_name = model_name, arima_obj = arima_obj)

        else:
            raise NotImplementedError("Not Implemented. Currently we support only (SARIMAX , SARIMAXResultsWrapper) , (ARIMA , ARIMAResultsWrapper) Combinations of Model and Result Objects.")
