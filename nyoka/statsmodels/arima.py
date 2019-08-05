from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from pprint import pprint
from PMML44 import *
from datetime import datetime
import metadata
import warnings
import math

class ArimaToPMML:
    def __init__(self, time_series_data=None, model_obj=None, results_obj=None, pmml_file_name="from_arima.pmml"):
        """
        Write a PMML file using model-object, model-parameters and time series data. Models are built using Statsmodels.

        Parameters:
        -----------
        time_series_data: (Optional)
            Pandas Series object
        model_obj: (Optional)
            Instance of ARIMA/SARIMAX from statsmodels
        results_obj: 
            Instance of ARIMAResultsWrapper/SARIMAXResultsWrapper from statsmodels
        pmml_file_name: string
        """

        def ExportToPMML(model_name = None, arima_obj = None):
            n_columns = 1  # because we are dealing with Series object
            n_samples = results_obj.model.nobs
            function_name = 'timeSeries'
            best_fit = 'ARIMA'

            pmml = PMML(
                version = '4.4',
                Header = Header(copyright = "Copyright (c) 2018 Software AG", description = "ARIMA Model",  
                                Timestamp = Timestamp(datetime.utcnow()),
                                Application=Application(name="Nyoka",version=metadata.__version__)),
                DataDictionary = DataDictionary(numberOfFields = n_columns, 
                                                DataField = get_data_field_objs()),
                TimeSeriesModel = [ TimeSeriesModel(modelName = model_name,
                                                    functionName = function_name, bestFit = best_fit, isScorable = True,
                                                    MiningSchema = MiningSchema(MiningField = get_mining_field_objs()),
                                                    TimeSeries = get_time_series_obj_list(usage = 'original' , timeRequired = True),
                                                    ARIMA = arima_obj) ])

            pmml.export(outfile=open(pmml_file_name, "w"), level=0)

        def get_sarimax_obj(sm_results):
            #NonSeasonal
            p = sm_results._results.specification.k_ar
            if 'k_diff' in sm_results._results.specification.__dict__.keys():
                d = sm_results._results.specification.k_diff
            else:
                d = 0
            q = sm_results._results.specification.k_ma
            
            ns_ar_content = ' '.join([str(i) for i in sm_results._results._params_ar] if int(p) > 0 else [])
            ns_ar_params_array = ArrayType(content = ns_ar_content, n = p, type_ = 'real')

            ns_ma_content = ' '.join([str(coeff) for coeff in sm_results._results._params_ma] if int(q) > 0 else [])
            ns_ma_coeff_array = ArrayType(content = ns_ma_content, n = q, type_ = 'real')
            ny_ns_maCoef_obj = MACoefficients(Array = ns_ma_coeff_array)

            #Seasonal
            P = sm_results._results.specification.seasonal_order[0]
            D = sm_results._results.specification.seasonal_order[1]
            Q = sm_results._results.specification.seasonal_order[2]
            S = sm_results._results.specification.seasonal_periods

            seasonal_ar_content = ' '.join([str(i) for i in sm_results._results._params_seasonal_ar] if int(P) > 0 else [])
            seasonal_ar_params_array = ArrayType(content = seasonal_ar_content, n = P, type_ = 'real')

            seasonal_ma_content = ' '.join([str(coeff) for coeff in sm_results._results._params_seasonal_ma] if int(Q) > 0 else [])
            seasonal_ma_coeff_array = ArrayType(content = seasonal_ma_content, n = Q, type_ = 'real')
            ny_seasonal_maCoef_obj = MACoefficients(Array = seasonal_ma_coeff_array)

            nyoka_sarimax_obj = ARIMA(#predictionMethod = None,
                                Extension = get_sarimax_extension_list(sm_results),
                                NonseasonalComponent = NonseasonalComponent(p = p, d = d, q = q, AR = AR(Array = ns_ar_params_array), MA = MA(MACoefficients = ny_ns_maCoef_obj)),
                                SeasonalComponent = SeasonalComponent(P = P, D = D, Q = Q, period = S, AR = AR(Array = seasonal_ar_params_array), MA = MA(MACoefficients = ny_seasonal_maCoef_obj)))
            return nyoka_sarimax_obj

        def get_arima_obj(sm_results):
            p = sm_results._results.k_ar
            if 'k_diff' in sm_results._results.__dict__.keys():
                d = sm_results._results.k_diff
            else:
                d = 0
            q = sm_results._results.k_ma
            if sm_results.model.method == 'css':
                pred_method = "conditionalLeastSquares"
            elif sm_results.model.method in ['mle', 'css-mle']:
                pred_method = "exactLeastSquares"

            ar_content = ' '.join([str(i) for i in sm_results._results.arparams] if int(p) > 0 else [])
            ar_params_array = ArrayType(content = ar_content, n = p, type_ = 'real')
            
            ma_content = ' '.join([str(coeff) for coeff in sm_results._results.maparams] if int(q) > 0 else [])
            ma_coeff_array = ArrayType(content = ma_content, n = q, type_ = 'real')
            ny_maCoef_obj = MACoefficients(Array = ma_coeff_array)

            residuals = list()
            try:
                residuals = sm_results._results.resid[-q:]
            except:
                pass
            resid_content = ' '.join([str(res) for res in residuals])
            resid_array = ArrayType(content = resid_content, n = len(residuals), type_ = 'real')
            residual_obj = Residuals(Array = resid_array)


            const_term = 0
            if sm_results.k_trend:
                const_term = sm_results.params[0]

            nyoka_arima_obj = ARIMA(constantTerm = const_term,
                                predictionMethod = pred_method,
                                RMSE=math.sqrt(sm_results.model.sigma2),
                                NonseasonalComponent = NonseasonalComponent(p = p, d = d, q = q, AR = AR(Array = ar_params_array),\
                                      MA = MA(MACoefficients = ny_maCoef_obj, Residuals = residual_obj))
                                )
            return nyoka_arima_obj

        def get_time_series_obj_list(usage = 'original' , timeRequired = True):
            get_time_series_obj_list = list()
            def get_time_value_objs():
                time_value_objs = list()
                if(usage == 'original' and timeRequired == True):
                    for int_idx in range(results_obj.data.endog.size):
                        time_value_objs.append(TimeValue(index = int_idx, value = str(results_obj.data.endog[int_idx]), Timestamp = Timestamp(str(results_obj.model.data.orig_endog._index[int_idx]))))
                elif(usage == 'logical' and timeRequired == True):
                    #TODO: Implement This
                    raise NotImplementedError("Not Implemented")
                return time_value_objs

            obj = TimeSeries(usage = usage, startTime = 0, endTime = results_obj.data.endog.size - 1, interpolationMethod = 'none', TimeValue = get_time_value_objs())
            get_time_series_obj_list.append(obj)
            return get_time_series_obj_list

        def get_mining_field_objs():
            mining_field_objs = list()
            idx_name = results_obj.data.orig_endog.index.name
            idx_usage_type = 'order'
            mining_field_objs.append(MiningField(name = idx_name, usageType = idx_usage_type))
            if results_obj.data.orig_endog.__class__.__name__ == 'DataFrame':
                ts_name = results_obj.data.orig_endog.columns[0]
            elif results_obj.data.orig_endog.__class__.__name__ == 'Series':
                ts_name = results_obj.data.orig_endog.name
            else:
                ts_name = 'input'
            ts_usage_type = 'target'
            mining_field_objs.append(MiningField(name = ts_name, usageType = ts_usage_type))
            mining_field_objs.append(MiningField(name = 'h', usageType = 'supplementary'))
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

        def get_data_field_objs():
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
            tar_data_type, ts_op_type = get_pmml_datatype_optype(results_obj.model.endog)
            data_field_objs.append(DataField(name=ts_name, dataType=tar_data_type, optype=ts_op_type))
            data_field_objs.append(DataField(name='h', dataType='integer', optype='continuous'))
            return data_field_objs

        def get_sarimax_extension_list(results):
            extensions = list()
            extensions.append(Extension(name="sigmaSquare", value = results._params_variance[0]))
            extensions.append(Extension(name="cov_type", value = results.cov_type))
            return extensions

        if results_obj.model.__class__.__name__ == 'SARIMAX':
            #Get SArimaX Object and Export
            sarimax_obj = get_sarimax_obj(results_obj)
            model_name = 'SARIMAX'
            ExportToPMML(model_name = model_name, arima_obj = sarimax_obj)

        elif results_obj.model.__class__.__name__ in ['ARIMA', 'ARMA']:
            #Get Arima Object and Export
            arima_obj = get_arima_obj(results_obj)
            model_name = 'ARIMA'
            ExportToPMML(model_name = model_name, arima_obj = arima_obj)

        else:
            raise NotImplementedError("Not Implemented. Currently we support only ARIMA and SARIMAX.")
