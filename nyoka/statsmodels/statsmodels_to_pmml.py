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


from pprint import pprint
from PMML44 import *
from datetime import datetime
import warnings
import math
from base.constants import *

class StatsmodelsToPmml:
    """
    Exports time-series models from statsmodels library into PMML.

    Parameters:
    -----------
    results_obj: 
        Instance of AR(I)MAResultsWrapper / (SARI/VAR)MAXResultsWrapper from statsmodels
    pmml_file_name: string
        Name of the PMML
    conf_int : list (optional)
        Confidence intervel. A list of values mentioning the percentage of confidence.
        e.g., conf_int = [80,95] will create OutputField for lower bound and upper bound of confidence interval with 80% and 95%.
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description of the model
    Returns
    -------
    Generates PMML object and exports it to `pmml_file_name`
    """
    def __init__(self, results_obj=None, pmml_file_name="from_statsmodels.pmml", conf_int=None, model_name=None, description=None):
        self.results_obj = results_obj
        self.pmml_file_name = pmml_file_name
        self.conf_int = conf_int
        self.model_name = model_name
        self.description = description
        self.pmml = None
        self.construct_pmml()
        self.export_pmml()

    def export_pmml(self):
        """
        Writes the generated PMML object to given `pmml_file_name`
        """
        pmml = PMML(
            version=PMML_SCHEMA.VERSION,
            Header=Header(
                copyright = HEADER_INFO.COPYRIGHT,
                description = self.description if self.description else HEADER_INFO.DEFAULT_DESCRIPTION,
                Timestamp = Timestamp(datetime.now()),
                Application=Application(name=HEADER_INFO.APPLICATION_NAME,version=HEADER_INFO.APPLICATION_VERSION)
            ),
            DataDictionary=self.data_dictionary,
            TimeSeriesModel=[self.ts_model]
        )
        pmml.export(open(self.pmml_file_name,'w'),0)

    def generate_data_dictionary(self):
        """
        Generates DataDictionary Object. The number of DataField is one more than the dimension of the data.\
        The extra DataField is a supplementary to hold the value of `h`(horizon) for forecasting.
        """
        data_fields = []
        for val in self.y:
            data_fields.append(
                DataField(
                    name=val,
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE
                )
            )
        data_fields.append(
            DataField(name="h", optype=OPTYPE.CONTINUOUS, dataType=DATATYPE.INTEGER)
        )
        self.data_dictionary = DataDictionary(
            numberOfFields=len(data_fields),
            DataField=data_fields
        )

    def construct_pmml(self):
        """
        Constructs the actual model object. (ARIMA/ TimeSeriesModel)
        """
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
            self.model_name = self.model_name if self.model_name else "ArimaModel"
            self.description = self.description if self.description else "Non-Seasonal Arima Model"
            if hasattr(self.results_obj,"fit_details"):
                best_fit = TIMESERIES_ALGORITHM.STATE_SPACE_MODEL
                state_space_model = self.generate_state_space_model()
            else:
                best_fit = TIMESERIES_ALGORITHM.ARIMA
                arima_model = self.generate_arima_model()
        elif self.model.__class__.__name__ in ['VARMAX','SARIMAX']:
            best_fit = TIMESERIES_ALGORITHM.STATE_SPACE_MODEL
            self.model_name = self.model_name if self.model_name else self.model.__class__.__name__
            self.description = self.description if self.description else "State Space Model"
            state_space_model = self.generate_state_space_model()
        else:
            raise NotImplementedError("Not Implemented. Currently we support only ARMA, ARIMA, SARIMAX and VARMAX.")

        self.ts_model = TimeSeriesModel(
            modelName=self.model_name,
            functionName=MINING_FUNCTION.TIMESERIES,
            bestFit=best_fit,
            MiningSchema=mining_schema,
            Output=output,
            TimeSeries=time_series_list,
            ARIMA=arima_model,
            StateSpaceModel=state_space_model
        )


    def generate_state_space_model(self):
        """
        Constructs StateSpaceModel object. For the following models -\
        - `statsmodels.tsa.statespace.sarimax.SARIMAX`
        - `statsmodels.tsa.statespace.varmax.VARMAX`
        - `statsmodels.tsa.statespace.tsa.arima.ARIMA`
        """
        import numpy as np
        np.set_printoptions(precision=12)
        selected_state_cov_matrix = None
        predicted_state_cov_matrix = None
        observation_cov_matrix = None
        smoother_results = self.results_obj.smoother_results
        S_t0 = self.results_obj.filtered_state[...,-1]
        F_matrix = smoother_results.transition[...,-1] #transition_matrix
        G = smoother_results.design[...,-1] #measurement_matrix

        if self.model.__class__.__name__ in ["ARMA","ARIMA"]:
            intercept = [smoother_results.obs_intercept[...,-1][0]]
            intercept_type = "observation"
            S_t1 = np.dot(F_matrix, S_t0)
        else:
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
            intercept_type = "state"
            S_t1 = np.dot(F_matrix, S_t0) + intercept
        arr_content = " ".join([str(val) for val in intercept])
        arr = ArrayType(type_=ARRAY_TYPE.REAL,content=arr_content, n=len(intercept))
        intercept_vector = InterceptVector(Array=arr, type_=intercept_type)

        t_mat = Matrix(nbRows=F_matrix.shape[0], nbCols=F_matrix.shape[1])
        for row in F_matrix:
            array_content = " ".join([str(val) for val in row])
            t_mat.add_Array(ArrayType(content=array_content, type_=ARRAY_TYPE.REAL))
        transition_matrix = TransitionMatrix(Matrix=t_mat)

        m_mat = Matrix(nbRows=G.shape[0], nbCols=G.shape[1])
        for row in G:
            array_content = " ".join([str(val) for val in row])
            m_mat.add_Array(ArrayType(content=array_content, type_=ARRAY_TYPE.REAL))
        measurement_matrix = MeasurementMatrix(Matrix=m_mat)

        arr_content = " ".join([str(val) for val in S_t1])
        arr = ArrayType(type_=ARRAY_TYPE.REAL,content=arr_content, n=len(S_t1))
        final_state_vector = FinalStateVector(Array=arr)


        #For confidence interval
        if self.conf_int is not None:
            R = smoother_results.selection[...,-1] # selection matrix
            Q = smoother_results.state_cov[...,-1] # state_covariance matrix
            obs_cov = smoother_results.obs_cov[..., -1] # observation covariance
            R_Q_R_prime = np.dot(R,np.dot(Q,R.T)) # selected_state_cov
            P_t0 = smoother_results.predicted_state_cov[...,-1] # predicted_state_cov

            RQR_mat = Matrix(nbRows=R_Q_R_prime.shape[0], nbCols=R_Q_R_prime.shape[1])
            for row in R_Q_R_prime:
                array_content = " ".join([str(val) for val in row])
                RQR_mat.add_Array(ArrayType(content=array_content, type_=ARRAY_TYPE.REAL))
            selected_state_cov_matrix = SelectedStateCovarianceMatrix(Matrix=RQR_mat)
            p_mat = Matrix(nbRows=P_t0.shape[0], nbCols=P_t0.shape[1])
            for row in P_t0:
                array_content = " ".join([str(val) for val in row])
                p_mat.add_Array(ArrayType(content=array_content, type_=ARRAY_TYPE.REAL))
            predicted_state_cov_matrix = PredictedStateCovarianceMatrix(Matrix=p_mat)
            h_mat = Matrix(nbRows=obs_cov.shape[0], nbCols=obs_cov.shape[1])
            for row in obs_cov:
                array_content = " ".join([str(val) for val in row])
                h_mat.add_Array(ArrayType(content=array_content, type_=ARRAY_TYPE.REAL))
            observation_cov_matrix = ObservationVarianceMatrix(Matrix=h_mat)

        state_space_model = StateSpaceModel(
            StateVector=final_state_vector,
            TransitionMatrix=transition_matrix,
            MeasurementMatrix=measurement_matrix,
            InterceptVector=intercept_vector,
            SelectedStateCovarianceMatrix=selected_state_cov_matrix,
            PredictedStateCovarianceMatrix=predicted_state_cov_matrix,
            ObservationVarianceMatrix=observation_cov_matrix
        )
        return state_space_model


    def generate_arima_model(self):
        """
        Constructs ARIMA object. Only for `statsmodels.tsa.arima_model.ARIMA` class.
        """
        p = self.results_obj.k_ar
        q = self.results_obj.k_ma
        d = getattr(self.results_obj,'k_diff',0)

        ar = None
        ma = None
        if p > 0:
            ar_content = ' '.join([str(i) for i in self.results_obj.arparams])
            ar_params_array = ArrayType(content = ar_content, n = p, type_ = ARRAY_TYPE.REAL)
            ar = AR(Array = ar_params_array)
        if q > 0:
            ma_content = ' '.join([str(coeff) for coeff in self.results_obj.maparams])
            ma_coeff_array = ArrayType(content = ma_content, n = q, type_ = ARRAY_TYPE.REAL)
            ny_maCoef_obj = MACoefficients(Array = ma_coeff_array)

            residuals = self.results_obj.resid[-q:]
            resid_content = ' '.join([str(res) for res in residuals])
            resid_array = ArrayType(content = resid_content, n = q, type_ = ARRAY_TYPE.REAL)
            residual_obj = Residuals(Array = resid_array)
            ma = MA(MACoefficients = ny_maCoef_obj, Residuals = residual_obj)

        const_term = 0
        if self.results_obj.k_trend:
            const_term = self.results_obj.params[0]
        non_seasonal_comp = NonseasonalComponent(p = p, d = d, q = q, AR = ar, MA = ma)

        rmse = math.sqrt(self.model.sigma2)

        arima_obj = ARIMA(constantTerm = const_term,
                                predictionMethod = ARIMA_PREDICTION_METHOD.CSS,
                                RMSE=rmse,
                                NonseasonalComponent = non_seasonal_comp
                                )
        return arima_obj


    def generate_time_value_object(self, data):
        """
        Generates TimeValue object. If data has any index, then the index will be in TimeStamp object.
        """
        time_values = []
        indices = self.data_obj.dates
        for data_idx in range(len(data)):
            tv = TimeValue(index=data_idx,value=data[data_idx],\
                Timestamp=Timestamp(str(indices[data_idx])) if indices is not None else None)
            time_values.append(tv)
        return time_values

    def generate_time_series(self):
        """
        Generates TimeSeries object. The number of TimeSeries object is equal to the dimeansion of the data.
        """
        time_series_list = []
        if self.data_obj.endog.ndim == 1:
            ts = TimeSeries(usage = TIMESERIES_USAGE.ORIGINAL, field=self.y[0], startTime = 0,\
                 endTime = len(self.data_obj.endog) - 1,\
                 TimeValue = self.generate_time_value_object(self.data_obj.endog))
            time_series_list.append(ts)
        else:
            for i in range(self.data_obj.endog.shape[-1]):
                ts = TimeSeries(usage = TIMESERIES_USAGE.ORIGINAL, field=self.y[i], startTime = 0,\
                     endTime = len(self.data_obj.endog) - 1,\
                 TimeValue = self.generate_time_value_object(self.data_obj.endog[...,i]))
                time_series_list.append(ts)
        return time_series_list


    def generate_output(self):
        """
        Generates Output object. If user provides value in `conf_int` parameter, then there will be two OuputField\
        for each value. One with `feature=confidenceIntervalLower` and another with `feature=confidenceIntervalUpper`.
        """
        out_flds = []
        for y_ in self.y:
            out_flds.append(
                OutputField(
                    name="predicted_"+y_,
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.STRING,
                    feature=RESULT_FEATURE.PREDICTED_VALUE,
                    Extension=[Extension(extender="ADAPA",name="dataType",value="json")]
                    )
            )
        if self.conf_int is not None:
            lower = []
            upper = []
            for percent in self.conf_int:
                for y_ in self.y:
                    lower.append(
                        OutputField(
                            name=f"conf_int_{percent}_lower_{y_}",
                            optype=OPTYPE.CONTINUOUS,
                            dataType=DATATYPE.STRING,
                            targetField=y_,
                            feature=RESULT_FEATURE.CONFIDENCE_INTERVAL_LOWER,
                            value=percent,
                            Extension=[Extension(extender="ADAPA",name="dataType",value="json")]
                            )
                    )
                    upper.append(
                        OutputField(
                            name=f"conf_int_{percent}_upper_{y_}",
                            optype=OPTYPE.CONTINUOUS,
                            dataType=DATATYPE.STRING,
                            targetField=y_,
                            feature=RESULT_FEATURE.CONFIDENCE_INTERVAL_UPPER,
                            value=percent,
                            Extension=[Extension(extender="ADAPA",name="dataType",value="json")]
                        )
                    )
            out_flds.extend(lower + upper)
        return Output(OutputField=out_flds)

    def generate_mining_schema(self):
        """
        Generates MiningSchema object.
        """
        mining_fields = []
        for y_ in self.y:
            mining_fields.append(MiningField(name = y_, usageType = FIELD_USAGE_TYPE.TARGET))
        mining_fields.append(MiningField(name = 'h', usageType = FIELD_USAGE_TYPE.SUPPLEMENTARY))
        return MiningSchema(MiningField=mining_fields)