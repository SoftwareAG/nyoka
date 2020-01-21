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
    """
    Write a PMML file using model-object, model-parameters and time series data. Models are built using Statsmodels.

    Parameters:
    -----------
    results_obj: 
        Instance of ARIMAResultsWrapper/SARIMAXResultsWrapper from statsmodels
    pmml_file_name: string
        Name of the PMML
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description of the model
    Returns
    -------
    Generates PMML object and exports it to `pmml_file_name`
    """
    def __init__(self, results_obj=None, pmml_file_name="from_arima.pmml", model_name=None, description=None):
        self.results_obj = results_obj
        self.pmml_file_name = pmml_file_name
        self.model_name = model_name
        self.description = description
        self.construct_pmml()
        self.export_pmml()

    def export_pmml(self):
        pmml = PMML(
            version="4.4",
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
                    optype="continuous",
                    dataType="double"
                )
            )
        data_fields.append(
            DataField(name="h", optype="continous", dataType="integer")
        )
        self.data_dictionary = DataDictionary(
            numberOfFields=len(self.y)+1,
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
            self.y = [self.y]
        self.generate_data_dictionary()

        output = self.generate_output()
        mining_schema = self.generate_mining_schema()
        time_series_list = self.generate_time_series()
        arima_model = None
        state_space_model = None
        best_fit = "ARIMA"

        if self.model.__class__.__name__ in ['ARMA', 'ARIMA']:
            self.model_name = self.model_name if self.model_name else "ArimaModel"
            self.description = self.description if self.description else "Non-Seasonal Arima Model"
            arima_model = self.generate_arima_model()
        elif self.model.__class__.__name__ == 'SARIMAX':
            self.model_name = self.model_name if self.model_name else "SarimaxModel"
            self.description = self.description if self.description else "Seasonal Arima Model"
            arima_model = self.generate_seasonal_arima_model()
        elif self.model.__class__.__name__ == "VARMAX":
            best_fit = "StateSpaceModel"
            self.model_name = self.model_name if self.model_name else "VarmaxModel"
            self.description = self.description if self.description else "Vector Arma Model"
            state_space_model = self.generate_state_space_model()
        else:
            raise NotImplementedError("Not Implemented. Currently we support only ARMA, ARIMA, SARIMAX and VARMAX.")

        self.ts_model = TimeSeriesModel(
            modelName=self.model_name,
            functionName="timeSeries",
            bestFit=best_fit,
            MiningSchema=mining_schema,
            Output=output,
            TimeSeries=time_series_list,
            ARIMA=arima_model,
            StateSpaceModel=state_space_model
        )


    def generate_state_space_model(self):
        import numpy as np
        smoother_results = self.results_obj.smoother_results
        S_t0 = smoother_results.filtered_state[:,-1]

        mu = None
        intercept_vector = None
        if self.model.k_trend:
            mu=self.results_obj.params[self.model._params_trend]
            if mu.__class__.__name__ == 'Series':
                mu = mu.values

        F_matrix = smoother_results.transition
        F_matrix = np.reshape(F_matrix, F_matrix.shape[:-1]) #transition_matrix

        G = smoother_results.design
        G = np.reshape(G, G.shape[:-1]) #measurement_matrix

        S_t1 = np.dot(F_matrix, S_t0) #finalStateVector

        t_mat = Matrix(nbRows=F_matrix.shape[0], nbCols=F_matrix.shape[1])
        for row in F_matrix:
            array_content = []
            for col in row:
                array_content.append(str(col))
            array_content = " ".join(array_content)
            t_mat.add_Array(ArrayType(content=array_content, type_='real'))
        transition_matrix = TransitionMatrix(Matrix=t_mat)

        m_mat = Matrix(nbRows=G.shape[0], nbCols=G.shape[1])
        for row in G:
            array_content = []
            for col in row:
                array_content.append(str(col))
            array_content = " ".join(array_content)
            m_mat.add_Array(ArrayType(content=array_content, type_='real'))
        measurement_matrix = MeasurementMatrix(Matrix=m_mat)

        arr_content = []
        for val in S_t1:
            arr_content.append(str(val))
        arr_content = " ".join(arr_content)
        arr = ArrayType(type_='real',content=arr_content, n=len(S_t1))
        final_state_vector = FinalStateVector(Array=arr)

        if mu is not None:
            arr_content = []
            for val in mu:
                arr_content.append(str(val))
            arr_content = " ".join(arr_content)
            arr = ArrayType(type_='real',content=arr_content, n=len(mu))
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
        
        pred_method = "conditionalLeastSquares"
        ar = None
        ma = None
        if p > 0:
            ar_content = ' '.join([str(i) for i in self.results_obj.arparams])
            ar_params_array = ArrayType(content = ar_content, n = p, type_ = 'real')
            ar = AR(Array = ar_params_array)
        if q > 0:
            ma_content = ' '.join([str(coeff) for coeff in self.results_obj.maparams])
            ma_coeff_array = ArrayType(content = ma_content, n = q, type_ = 'real')
            ny_maCoef_obj = MACoefficients(Array = ma_coeff_array)

            residuals = self.results_obj.resid[-q:] if q>0 else []
            resid_content = ' '.join([str(res) for res in residuals])
            resid_array = ArrayType(content = resid_content, n = len(residuals), type_ = 'real')
            residual_obj = Residuals(Array = resid_array)
            ma = MA(MACoefficients = ny_maCoef_obj, Residuals = residual_obj)

        const_term = 0
        if self.results_obj.k_trend:
            const_term = self.results_obj.params[0]
        non_seasonal_comp = NonseasonalComponent(p = p, d = d, q = q, AR = ar, MA = ma)

        rmse = math.sqrt(self.model.sigma2)

        arima_obj = ARIMA(constantTerm = const_term,
                                predictionMethod = pred_method,
                                RMSE=rmse,
                                NonseasonalComponent = non_seasonal_comp
                                )
        return arima_obj

    def generate_seasonal_arima_model(self):
        import numpy as np
        # Non-Seasonal part
        p = self.results_obj.specification.k_ar
        d = getattr(self.results_obj.specification,"k_diff",0)
        q = self.results_obj.specification.k_ma
        ar = None
        ma = None
        if p > 0:
            ns_ar_content = ' '.join([str(i) for i in self.results_obj._params_ar])
            ns_ar_params_array = ArrayType(content = ns_ar_content, n = p, type_ = 'real')
            ar = AR(Array = ns_ar_params_array)
        if q < 0:
            ns_ma_content = ' '.join([str(coeff) for coeff in self.results_obj._params_ma])
            ns_ma_coeff_array = ArrayType(content = ns_ma_content, n = q, type_ = 'real')
            ny_ns_maCoef_obj = MACoefficients(Array = ns_ma_coeff_array)
            ma = MA(MACoefficients = ny_ns_maCoef_obj)

        non_seasonal_comp = NonseasonalComponent(p = p, d = d, q = q, AR = ar, MA = ma)

        constant_term = 0
        if self.results_obj.specification.k_trend >0:
            constant_term = self.results_obj._params_trend[0]

        rmse = math.sqrt(self.results_obj._params_variance[0])

        #Seasonal part
        P = self.results_obj.specification.seasonal_order[0]
        D = self.results_obj.specification.seasonal_order[1]
        Q = self.results_obj.specification.seasonal_order[2]
        S = self.results_obj.specification.seasonal_periods

        sar = None
        sma = None
        if P > 0:
            seasonal_ar_content = ' '.join([str(i) for i in self.results_obj._params_seasonal_ar])
            seasonal_ar_params_array = ArrayType(content = seasonal_ar_content, n = P, type_ = 'real')
            sar = AR(Array = seasonal_ar_params_array)
        if Q > 0:
            seasonal_ma_content = ' '.join([str(coeff) for coeff in self.results_obj._params_seasonal_ma])
            seasonal_ma_coeff_array = ArrayType(content = seasonal_ma_content, n = Q, type_ = 'real')
            ny_seasonal_maCoef_obj = MACoefficients(Array = seasonal_ma_coeff_array)
            sma = MA(MACoefficients = ny_seasonal_maCoef_obj)

        seasonal_comp = SeasonalComponent(P = P, D = D, Q = Q, period = S, AR = sar, MA = sma)

        #MaximumLikelihoodStat
        smoother_results = self.results_obj.smoother_results
        S_t0=smoother_results.filtered_state[:,-1]

        F_matrix= smoother_results.transition
        F_matrix = np.reshape(F_matrix, F_matrix.shape[:-1]) #transition_matrix

        G = smoother_results.design
        G = np.reshape(G, G.shape[:-1]) #measurement_matrix

        S_t1 = np.dot(F_matrix, S_t0) #finalStateVector

        t_mat = Matrix(nbRows=F_matrix.shape[0], nbCols=F_matrix.shape[1])
        for row in F_matrix:
            array_content = []
            for col in row:
                array_content.append(str(col))
            array_content = " ".join(array_content)
            t_mat.add_Array(ArrayType(content=array_content, type_='real'))
        transition_matrix = TransitionMatrix(Matrix=t_mat)

        m_mat = Matrix(nbRows=G.shape[0], nbCols=G.shape[1])
        for row in G:
            array_content = []
            for col in row:
                array_content.append(str(col))
            array_content = " ".join(array_content)
            m_mat.add_Array(ArrayType(content=array_content, type_='real'))
        measurement_matrix = MeasurementMatrix(Matrix=m_mat)

        arr_content = []
        for val in S_t1:
            arr_content.append(str(val))
        arr_content = " ".join(arr_content)
        arr = ArrayType(type_='real',content=arr_content, n=len(S_t1))
        finalStateVector = FinalStateVector(Array=arr)


        # fomega_mat = Matrix(kind="symmetric", nbRows=1, nbCols=1)
        # fomega_mat.add_Array(ArrayType(content="0", type_='real'))
        # finalOmega = FinalOmega(Matrix=fomega_mat)


        kalman_state = KalmanState(
            FinalStateVector=finalStateVector,
            TransitionMatrix=transition_matrix,
            MeasurementMatrix=measurement_matrix
            )
        max_lik_stat = MaximumLikelihoodStat(method='kalman',KalmanState=kalman_state)

        arima_obj = ARIMA(RMSE=rmse, constantTerm = constant_term,
                                predictionMethod = "exactLeastSquares",
                                NonseasonalComponent = non_seasonal_comp,
                                SeasonalComponent = seasonal_comp,
                                MaximumLikelihoodStat = max_lik_stat 
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
            ts = TimeSeries(usage = 'original', field=self.y[0], startTime = 0, endTime = len(self.data_obj.endog) - 1,\
                 interpolationMethod = 'none', TimeValue = self.generate_time_value_object(self.data_obj.endog))
            time_series_list.append(ts)
        else:
            for i in range(self.data_obj.endog.shape[-1]):
                ts = TimeSeries(usage = 'original', field=self.y[i], startTime = 0, endTime = len(self.data_obj.endog) - 1,\
                 interpolationMethod = 'none', TimeValue = self.generate_time_value_object(self.data_obj.endog[...,i]))
                time_series_list.append(ts)
        return time_series_list


    def generate_output(self):
        out_flds = []
        for y_ in self.y:
            out_flds.append(
                OutputField(
                    name="predicted_"+y_, 
                    optype="continuous",
                    dataType="double", 
                    feature="predictedValue"
                    )
            )
        if self.model.__class__.__name__ in ['ARIMA','ARMA']:
                names = ['cpi_80_lower','cpi_80_upper','cpi_95_lower','cpi_95_upper']
                values = ['LOWER80','UPPER80','LOWER95','UPPER95']
                for name, value in zip(names, values):
                    out_flds.append(OutputField(
                        name=name, 
                        optype='continuous', 
                        dataType='double',
                        feature='standardError',
                        Extension=[Extension(extender='ADAPA',name='cpi', value=value)]))
        return Output(OutputField=out_flds)
    
    def generate_mining_schema(self):
        mining_fields = []
        for y_ in self.y:
            mining_fields.append(MiningField(name = y_, usageType = 'target'))
        mining_fields.append(MiningField(name = 'h', usageType = 'supplementary'))
        return MiningSchema(MiningField=mining_fields)