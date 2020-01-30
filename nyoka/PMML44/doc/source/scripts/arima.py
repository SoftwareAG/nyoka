from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from pprint import pprint
from PMML44 import *
from datetime import datetime
import metadata

"""
Classes used in arima.py

"""

class ArimaToPMML:
    """
    Write a PMML file using model-object, model-parameters and time series data. Models are built using Statsmodels.

    Parameters:
    -----------
    time_series_data: Pandas Series object
        The input data
    model_obj: statsmodels model object
        Instance of ExponentialSmoothing from statsmodels
    results_obj: statsmodels trained model object
        Instance of HoltWintersResults from statsmodels
    pmml_file_name: string
        Name of the pmml
    """

    def ExportToPMML(self, model_name, arima_obj,ts_data,f_name):
        """
        Write the PMML file by aggregating all required data

        Parameters:
        -----------
        model_name: string
            Name of the model
        arima_obj: statsmodels model object
            ARIMA model object
        ts_data : Pandas Series object
            The input data
        f_name : string
            Pmml file name
        """
        n_columns = 1  # because we are dealing with Series object
        n_samples = ts_data.size
        function_name = 'timeSeries'
        best_fit = 'ARIMA'

        pmml = PMML(
            version = '4.4',
            Header = Header(copyright = "Copyright (c) 2018 Software AG", description = "ARIMA Model",  
                            Timestamp = Timestamp(datetime.utcnow()),
                            Application=Application(name="Nyoka",version=metadata.__version__)),
            DataDictionary = DataDictionary(numberOfFields = n_columns, 
                                            DataField = self.get_data_field_objs(ts_data)),
            TimeSeriesModel = [ TimeSeriesModel(modelName = model_name,
                                                functionName = function_name, bestFit = best_fit, isScorable = True,
                                                MiningSchema = MiningSchema(MiningField = self.get_mining_field_objs(ts_data)),
                                                TimeSeries = self.get_time_series_obj_list(ts_data, usage = 'original' , timeRequired = True),
                                                ARIMA = arima_obj) ])

        pmml.export(outfile=open(f_name, "w"), level=0)

    def get_sarimax_obj(self, sm_model, sm_results):
        """
        Create SeasonalArima's PMML object

        Parameters:
        -----------
        sm_model: statsmodels model object
            Statsmodels model object
        sm_results: statsmodels trained model object
            Statsmodels trained model object

        Returns:
        --------
        nyoka_sarimax_obj : Nyoka PMML44 Object
            ARIMA object

        """
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
                            Extension = self.get_sarimax_extension_list(sm_results),
                            NonseasonalComponent = NonseasonalComponent(p = p, d = d, q = q, AR = AR(Array = ns_ar_params_array), MA = MA(MACoefficients = ny_ns_maCoef_obj)),
                            SeasonalComponent = SeasonalComponent(P = P, D = D, Q = Q, period = S, AR = AR(Array = seasonal_ar_params_array), MA = MA(MACoefficients = ny_seasonal_maCoef_obj)))
        return nyoka_sarimax_obj

    def get_arima_obj(self, sm_model, sm_results):
        """
        Create Arima's PMML object

        Parameters:
        -----------
        sm_model: statsmodels model object
            Statsmodels model object
        sm_results: statsmodels trained model object
            Statsmodels trained model object

        Returns:
        --------
        nyoka_sarimax_obj : Nyoka PMML44 object
            ARIMA object

        """
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

        ar_content = ' '.join([str(i) for i in sm_results._results.arparams] if int(p) > 0 else [])
        ar_params_array = ArrayType(content = ar_content, n = p, type_ = 'real')
        
        ma_content = ' '.join([str(coeff) for coeff in sm_results._results.maparams] if int(q) > 0 else [])
        ma_coeff_array = ArrayType(content = ma_content, n = q, type_ = 'real')
        ny_maCoef_obj = MACoefficients(Array = ma_coeff_array)

        nyoka_arima_obj = ARIMA(constantTerm = sm_results.params[0],
                            predictionMethod = pred_method,
                            Extension = self.get_arima_extension_list(sm_model),
                            NonseasonalComponent = NonseasonalComponent(p = p, d = d, q = q, AR = AR(Array = ar_params_array),  MA = MA(MACoefficients = ny_maCoef_obj)))
        return nyoka_arima_obj


    def get_time_series_obj_list(self, ts_data, usage = 'original' , timeRequired = True):
        """
        Create TimeSeries object for time series data

        Parameters:
        -----------
        ts_data : pandas Series
            The input data
        usage : string
            usage type (default="original")
        timeRequired : boolean
            (default=True)

        Returns:
        get_time_series_obj_list : list
            A list of TimeSeries object
        """
        get_time_series_obj_list = list()
        ts_int_index = range(len(ts_data))
        time_value_objs = list()
        if(usage == 'original' and timeRequired == True):
            for int_idx in ts_int_index:
                time_value_objs.append(TimeValue(index = int_idx, value = ts_data.iat[int_idx], Timestamp = Timestamp(ts_data.index[int_idx])))
        elif(usage == 'logical' and timeRequired == True):
            #TODO: Implement This
            raise NotImplementedError("Not Implemented")

        obj = TimeSeries(usage = usage, startTime = 0, endTime = ts_data.size - 1, interpolationMethod = 'none', TimeValue = time_value_objs)
        get_time_series_obj_list.append(obj)
        return get_time_series_obj_list

    def get_mining_field_objs(self, ts_data):
        """
        Create a list with instances of MiningField

        Parameters:
        -----------
        ts_data : pandas Series
            The input data

        Returns:
        --------
        mining_field_obj : list
            A list of MiningField object
        """
        mining_field_objs = list()
        idx_name = ts_data.index.name
        idx_usage_type = 'order'
        mining_field_objs.append(MiningField(name = idx_name, usageType = idx_usage_type))
        ts_name = ts_data.name
        ts_usage_type = 'target'
        mining_field_objs.append(MiningField(name = ts_name, usageType = ts_usage_type))
        return mining_field_objs

    def get_pmml_datatype_optype(self, series_obj):
        """
        Create dataType and opType for the model object

        Parameters:
        -----------
        series_obj: statsmodels model object
            Statsmodels model object

        Returns:
        --------
        pmml_data_type: string
            Data type
        pmml_op_type: string
            Optype
        """
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

    def get_data_field_objs(self, ts_data):
        """
        Create a list with instances of DataField

        Parameters:
        -----------
        ts_data: Pandas Series object
            The input data

        Returns:
        --------
        data_field_objs: list
            A list of DataField object
        """
        data_field_objs = list()
        index_name = ts_data.index.name
        idx_data_type, idx_op_type = self.get_pmml_datatype_optype(ts_data.index)
        data_field_objs.append(DataField(name=index_name, dataType=idx_data_type, optype=idx_op_type))
        ts_name = ts_data.name
        ts_data_type, ts_op_type = self.get_pmml_datatype_optype(ts_data)
        data_field_objs.append(DataField(name=ts_name, dataType=ts_data_type, optype=ts_op_type))
        return data_field_objs

    def get_sarimax_extension_list(self, results):
        """
        Create Extension for SARIMAX object

        Parameters:
        -----------
        results: statsmodels model object
            Statsmodels trained model

        Returns:
        --------
        extensions : list
            A list of Extension object
        """
        extensions = list()
        extensions.append(Extension(name="sigmaSquare", value = results._params_variance[0]))
        extensions.append(Extension(name="cov_type", value = results.cov_type))
        extensions.append(Extension(name="approx_complex_step", value = results._cov_approx_complex_step))
        extensions.append(Extension(name="approx_centered", value = results._cov_approx_centered))
        return extensions

    def get_arima_extension_list(self, model):
        """
        Create Extension for ARIMA object

        Parameters:
        -----------
        model: statsmodels model object
            Statsmodels model object

        Returns:
        --------
        extensions : list
            A list of Extension object
        """
        extensions = list()
        extensions.append(Extension(name="sigmaSquare", value = model.sigma2))
        return extensions

    def __init__(self, time_series_data, model_obj, results_obj, pmml_file_name):
        
        if(time_series_data.__class__.__name__ == 'DataFrame'):
            time_series_data = time_series_data.T.squeeze()

        if(model_obj.__class__.__name__ == 'SARIMAX' and results_obj.__class__.__name__ == 'SARIMAXResultsWrapper'):
            #Get SArimaX Object and Export
            sarimax_obj = self.get_sarimax_obj(model_obj, results_obj)
            model_name = 'sarimax'
            self.ExportToPMML(model_name = model_name, arima_obj = sarimax_obj, ts_data=time_series_data, f_name=pmml_file_name)

        elif(model_obj.__class__.__name__ == 'ARIMA' and results_obj.__class__.__name__ == 'ARIMAResultsWrapper'):
            #Get Arima Object and Export
            arima_obj = self.get_arima_obj(model_obj, results_obj)
            model_name = 'arima'
            self.ExportToPMML(model_name = model_name, arima_obj = arima_obj, ts_data=time_series_data, f_name=pmml_file_name)

        else:
            raise NotImplementedError("Not Implemented. Currently we support only (SARIMAX , SARIMAXResultsWrapper) , (ARIMA , ARIMAResultsWrapper) Combinations of Model and Result Objects.")
