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
import numpy as np
from base.enums import * 

class ArimaToPMML:
    """
    Write a PMML file using model-object, model-parameters and time series data. Models are built using Statsmodels.

    Parameters:
    -----------
    results_obj: 
        Instance of AR(I)MAResultsWrapper / (SARI/VAR)MAXResultsWrapper from statsmodels
    pmml_file_name: string
        Name of the PMML
    cpi : list (optional)
        Confidence of prediction intervel. A list of values mentioning the percentage of confidence.
        e.g., cpi = [80.5,95] will create OutputField for lower bound and upper bound of confidence interval with 80.5% and 95%.
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description of the model
    Returns
    -------
    Generates PMML object and exports it to `pmml_file_name`
    """
    def __init__(self, results_obj=None, pmml_file_name="from_arima.pmml", cpi=None, model_name=None, description=None):
        self.results_obj = results_obj
        self.pmml_file_name = pmml_file_name
        self.cpi = cpi
        self.model_name = model_name
        self.description = description
        self.construct_pmml()
        self.export_pmml()

    def export_pmml(self):
        pmml = PMML(
            version=PMML_SCHEMA.VERSION.value,
            Header=Header(
                copyright = "Copyright (c) 2018 Software AG",
                description = self.description,
                Timestamp = Timestamp(datetime.now()),
                Application=Application(name="Nyoka",version=metadata.__version__)
            ),
            DataDictionary=self.data_dictionary,
            TimeSeriesModel=[self.ts_model]
        )
        pmml.export(open(self.pmml_file_name,'w'),0)

    def generate_data_dictionary(self):
        data_fields = []
        for val in self.y:
            data_fields.append(
                DataField(
                    name=val,
                    optype=OPTYPE.CONTINUOUS.value,
                    dataType=DATATYPE.DOUBLE.value
                )
            )
        data_fields.append(
            DataField(name="h", optype=OPTYPE.CONTINUOUS.value, dataType=DATATYPE.INTEGER.value)
        )
        self.data_dictionary = DataDictionary(
            numberOfFields=len(data_fields),
            DataField=data_fields
        )

    def construct_pmml(self):

        if 'int' in str(self.results_obj.model.endog.dtype):
            self.results_obj.model.endog=self.results_obj.model.endog.astype('float64')
            self.results_obj.model.data.endog=self.results_obj.model.data.endog.astype('float64')

        self.data_obj = self.results_obj.data
        self.model = self.results_obj.model
        self.y = self.results_obj.data.ynames
        if self.y.__class__.__name__ == "str":
            self.y = [self.y.split(".")[-1]]
        self.generate_data_dictionary()

        output = self.generate_output()
        mining_schema = self.generate_mining_schema()
        time_series_list = self.generate_time_series()
        arima_model = None
        state_space_model = None

        if self.model.__class__.__name__ in ['ARMA', 'ARIMA']:
            best_fit = TIMESERIES_ALGORITHM.ARIMA.value
            self.model_name = self.model_name if self.model_name else "ArimaModel"
            self.description = self.description if self.description else "Non-Seasonal Arima Model"
            arima_model = self.generate_arima_model()
        elif self.model.__class__.__name__ in ['VARMAX','SARIMAX']:
            best_fit = TIMESERIES_ALGORITHM.STATE_SPACE_MODEL.value
            self.model_name = self.model_name if self.model_name else self.model.__class__.__name__
            self.description = self.description if self.description else "State Space Model"
            state_space_model = self.generate_state_space_model()
        else:
            raise NotImplementedError("Not Implemented. Currently we support only ARMA, ARIMA, SARIMAX and VARMAX.")

        self.ts_model = TimeSeriesModel(
            modelName=self.model_name,
            functionName=MINING_FUNCTION.TIMESERIES.value,
            bestFit=best_fit,
            MiningSchema=mining_schema,
            Output=output,
            TimeSeries=time_series_list,
            ARIMA=arima_model,
            StateSpaceModel=state_space_model
        )


    def generate_state_space_model(self):
        smoother_results = self.results_obj.smoother_results
        S_t0 = self.results_obj.filtered_state[...,-1]

        #state_intercept might contain `nan` values. So here it is generated manually.
        intercept = np.zeros(S_t0.shape)
        if self.model.k_trend:
            if self.model.__class__.__name__ == 'VARMAX':
                mu = self.results_obj.params[self.model._params_trend]
                if mu.__class__.__name__ == 'Series':
                    mu = mu.values
                intercept[:len(mu)] += mu
            else:
                mu=self.results_obj._params_trend[0]
                if mu.__class__.__name__ == 'Series':
                    mu = mu.values
                spec = self.results_obj.specification
                k_state = spec['k_diff']+spec['seasonal_periods']*spec['k_seasonal_diff']
                intercept[k_state] += mu  

        F_matrix = smoother_results.transition[...,-1] #transition_matrix

        G = smoother_results.design[...,-1] #measurement_matrix

        S_t1 = np.dot(F_matrix, S_t0) + intercept #finalStateVector

        t_mat = Matrix(nbRows=F_matrix.shape[0], nbCols=F_matrix.shape[1])
        for row in F_matrix:
            array_content = " ".join([str(col) for col in row])
            t_mat.add_Array(ArrayType(content=array_content, type_=ARRAY_TYPE.REAL.value))
        transition_matrix = TransitionMatrix(Matrix=t_mat)

        m_mat = Matrix(nbRows=G.shape[0], nbCols=G.shape[1])
        for row in G:
            array_content = " ".join([str(col) for col in row])
            m_mat.add_Array(ArrayType(content=array_content, type_=ARRAY_TYPE.REAL.value))
        measurement_matrix = MeasurementMatrix(Matrix=m_mat)

        arr_content = " ".join([str(val) for val in S_t1])
        arr = ArrayType(type_=ARRAY_TYPE.REAL.value,content=arr_content, n=len(S_t1))
        final_state_vector = FinalStateVector(Array=arr)

        #InterceptVector will always present
        arr_content = " ".join([str(val) for val in intercept])
        arr = ArrayType(type_=ARRAY_TYPE.REAL.value,content=arr_content, n=len(intercept))
        intercept_vector = InterceptVector(Array=arr)

        state_space_model = StateSpaceModel(
            StateVector=final_state_vector,
            TransitionMatrix=transition_matrix,
            MeasurementMatrix=measurement_matrix,
            InterceptVector=intercept_vector
        )
        return state_space_model


    def generate_arima_model(self):
        p = self.results_obj.k_ar
        q = self.results_obj.k_ma
        d = getattr(self.results_obj,'k_diff',0)
        
        ar = None
        ma = None
        if p > 0:
            ar_content = ' '.join([str(i) for i in self.results_obj.arparams])
            ar_params_array = ArrayType(content = ar_content, n = p, type_ = ARRAY_TYPE.REAL.value)
            ar = AR(Array = ar_params_array)
        if q > 0:
            ma_content = ' '.join([str(coeff) for coeff in self.results_obj.maparams])
            ma_coeff_array = ArrayType(content = ma_content, n = q, type_ = ARRAY_TYPE.REAL.value)
            ny_maCoef_obj = MACoefficients(Array = ma_coeff_array)

            residuals = self.results_obj.resid[-q:]
            resid_content = ' '.join([str(res) for res in residuals])
            resid_array = ArrayType(content = resid_content, n = q, type_ = ARRAY_TYPE.REAL.value)
            residual_obj = Residuals(Array = resid_array)
            ma = MA(MACoefficients = ny_maCoef_obj, Residuals = residual_obj)

        const_term = 0
        if self.results_obj.k_trend:
            const_term = self.results_obj.params[0]
        non_seasonal_comp = NonseasonalComponent(p = p, d = d, q = q, AR = ar, MA = ma)

        rmse = math.sqrt(self.model.sigma2)

        arima_obj = ARIMA(constantTerm = const_term,
                                predictionMethod = ARIMA_PREDICTION_METHOD.CSS.value,
                                RMSE=rmse,
                                NonseasonalComponent = non_seasonal_comp
                                )
        return arima_obj

    
    def generate_time_value_object(self, data):
        time_values = []
        indices = self.data_obj.dates
        for data_idx in range(len(data)):
            tv = TimeValue(index=data_idx,value=data[data_idx],\
                Timestamp=Timestamp(str(indices[data_idx])) if indices is not None else None)
            time_values.append(tv)
        return time_values

    def generate_time_series(self):
        time_series_list = []
        if self.data_obj.endog.ndim == 1:
            ts = TimeSeries(usage = TIMESERIES_USAGE.ORIGINAL.value, field=self.y[0], startTime = 0,\
                 endTime = len(self.data_obj.endog) - 1,\
                 TimeValue = self.generate_time_value_object(self.data_obj.endog))
            time_series_list.append(ts)
        else:
            for i in range(self.data_obj.endog.shape[-1]):
                ts = TimeSeries(usage = TIMESERIES_USAGE.ORIGINAL.value, field=self.y[i], startTime = 0,\
                     endTime = len(self.data_obj.endog) - 1,\
                 TimeValue = self.generate_time_value_object(self.data_obj.endog[...,i]))
                time_series_list.append(ts)
        return time_series_list


    def generate_output(self):
        out_flds = []
        for y_ in self.y:
            out_flds.append(
                OutputField(
                    name="predicted_"+y_, 
                    optype=OPTYPE.CONTINUOUS.value,
                    dataType=DATATYPE.STRING.value, 
                    feature=RESULT_FEATURE.PREDICTED_VALUE.value,
                    Extension=[Extension(extender="ADAPA",name="dataType",value="json")]
                    )
            )
        if self.model.__class__.__name__ in ['ARIMA','ARMA'] and self.cpi is not None:
            for percent in self.cpi:
                out_flds.extend([
                    OutputField(
                        name=f"cpi_{percent}_lower",
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value,
                        feature=RESULT_FEATURE.CONFIDENCE_INTERVAL_LOWER.value,
                        value=percent
                        ),
                    OutputField(
                        name=f"cpi_{percent}_upper",
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value,
                        feature=RESULT_FEATURE.CONFIDENCE_INTERVAL_UPPER.value,
                        value=percent
                    )
                ])
        return Output(OutputField=out_flds)
    
    def generate_mining_schema(self):
        mining_fields = []
        for y_ in self.y:
            mining_fields.append(MiningField(name = y_, usageType = FIELD_USAGE_TYPE.TARGET.value))
        mining_fields.append(MiningField(name = 'h', usageType = FIELD_USAGE_TYPE.SUPPLEMENTARY.value))
        return MiningSchema(MiningField=mining_fields)