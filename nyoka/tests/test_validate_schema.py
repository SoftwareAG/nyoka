# Exporters
from nyoka import skl_to_pmml, StatsmodelsToPmml, ExponentialSmoothingToPMML

# Nyoka preprocessings
from nyoka.preprocessing import Lag

# Pipeline/ DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

# Sklearn Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, LabelEncoder,\
 Binarizer, PolynomialFeatures, LabelBinarizer
try:
    from sklearn.preprocessing import Imputer
except:
    from sklearn.impute import SimpleImputer as Imputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

## Sklearn models
# Linear models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

# Tree models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# SVM
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM

# Ensemble
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,\
     RandomForestRegressor, IsolationForest

# Clustering
from sklearn.cluster import KMeans

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# statsmodels models
from statsmodels.tsa.api import ARIMA, SARIMAX, VARMAX, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA as StateSpaceARIMA

import unittest
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
import xmlschema
import pandas as pd

class StatsmodelsDataHelper:
    def get_data_with_trend_and_seasonality(self):
        # data with trend and seasonality present
        # no of international visitors in Australia
        data = [41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015, 31.1802, 37.7179,
                40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668, 35.1919,
                44.3197, 47.9137]
        index = pd.date_range(start='2005', end='2010-Q4', freq='QS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_visitors'
        return ts_data


    def get_non_seasonal_data(self):
		# Non Seasonal Data
        data = [266,146,183,119,180,169,232,225,193,123,337,186,194,150,210,273,191,287,
                226,304,290,422,265,342,340,440,316,439,401,390,490,408,490,420,520,480]
        index = pd.date_range(start='2016-01-01', end='2018-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'date_index'
        ts_data.name = 'cars_sold'
        return ts_data

    def get_seasonal_data(self):
		# Seasonal Data
        data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150,
                178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235,
                229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315,
                364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467,
                404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407,
                362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432]
        index = pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_passengers'
        return ts_data

    def get_data_for_varmax(self):
        data = pd.read_csv("nyoka/tests/SanDiegoWeather.csv", parse_dates=True, index_col=0)
        return data
    

class PmmlValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Starting pmml validation tests.")
        data = datasets.load_iris()
        cls.X = data.data
        cls.y = data.target
        cls.y_bin = [i%2 for i in range(cls.X.shape[0])]
        cls.features = data.feature_names
        data = datasets.load_boston()
        cls.X_reg = data.data
        cls.y_reg = data.target
        cls.features_reg =  data.feature_names
        cls.schema = xmlschema.XMLSchema("nyoka/pmml44.xsd")
        cls.statsmodels_data_helper = StatsmodelsDataHelper()

    
    def test_validate_sklearn_linear_models_binary_class(self):
        model = LogisticRegression()
        pipe = Pipeline([
            ('sclaer', StandardScaler()),
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'linear_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)
   
    def test_validate_sklearn_linear_models_multiclass(self):
        df = pd.DataFrame(data=self.X, columns=self.features)
        df['species'] = self.y
        model = LogisticRegression()
        pipe = Pipeline([
            ('mapper', DataFrameMapper([
                (['sepal length (cm)'], Binarizer())
            ])),
            ('model',model)
        ])
        pipe.fit(df[self.features], df.species)
        file_name = 'linear_model_multi_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_linear_models_regression(self):
        model = LinearRegression()
        pipe = Pipeline([
            ('impute', Imputer()),
            ('feat', PolynomialFeatures()),
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'linear_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_tree_models_binary_class(self):
        model = DecisionTreeClassifier()
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'tree_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_tree_models_multiclass(self):
        model = DecisionTreeClassifier()
        pipe = Pipeline([
            ('pca', PCA()),
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'tree_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_tree_models_regression(self):
        model = DecisionTreeRegressor()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'tree_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_svm_models_binary_class(self):
        model = SVC()
        pipe = Pipeline([
            ('scaler',MaxAbsScaler()),
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'svm_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_svm_models_multiclass(self):
        model = SVC()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'svm_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_lda_models_binary_class(self):
        model = LinearDiscriminantAnalysis()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'lda_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_lda_models_multiclass(self):
        model = LinearDiscriminantAnalysis()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'lda_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)


    def test_validate_sklearn_mlp_models_multiclass(self):
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'mlp_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_svm_models_regression(self):
        model = SVR()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'svm_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_gboost_models_binary_class(self):
        model = GradientBoostingClassifier()
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'gboost_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_gboost_models_multiclass(self):
        df = pd.DataFrame(data=self.X, columns=self.features)
        df['new'] = [i%3 for i in range(self.X.shape[0])]
        df['species'] = self.y
        model = GradientBoostingClassifier()
        pipe = Pipeline([
            ('mapper', DataFrameMapper([
                ('new', LabelEncoder())
            ])),
            ('model',model)
        ])
        pipe.fit(df.drop(['species'],axis=1), df.species)
        file_name = 'gboost_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features+['new'], 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_gboost_models_regression(self):
        model = GradientBoostingRegressor()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'gboost_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_rf_models_binary_class(self):
        df = pd.DataFrame(data=self.X, columns=self.features)
        df['new'] = [i%3 for i in range(self.X.shape[0])]
        df['binary'] = self.y_bin
        model = RandomForestClassifier()
        pipe = Pipeline([
            ('mapper', DataFrameMapper([
                ('new', LabelBinarizer())
            ])),
            ('model',model)
        ])
        pipe.fit(df[self.features+['new']], df.binary)
        file_name = 'rf_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features+['new'], 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_rf_models_multiclass(self):
        model = RandomForestClassifier()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'rf_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_rf_models_regression(self):
        model = RandomForestRegressor()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'rf_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_linarsvr_models_regression(self):
        model = LinearSVR()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'linearsvr_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_gnb_models_binary_class(self):
        model = GaussianNB()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'gnb_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_gnb_models_multiclass(self):
        model = GaussianNB()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'gnb_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_knn_models_binary_class(self):
        model = KNeighborsClassifier()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'knn_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_linearSVC_models_multiclass(self):
        model = LinearSVC()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'linearsvc_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_knn_models_multiclass(self):
        model = KNeighborsClassifier()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'knn_model_numlti_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_knn_models_regression(self):
        model = KNeighborsRegressor()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'knn_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_kmeans_models(self):
        model = KMeans()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X)
        file_name = 'kmeans_model.pmml'
        skl_to_pmml(pipe, self.features, 'target',file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_sgd_with_text(self):
        categories = ['alt.atheism','talk.religion.misc']
        data = fetch_20newsgroups(subset='train', categories=categories)
        X = data.data[:4]
        Y = data.target[:4]
        features = ['input']
        target = 'output'
        model = SGDClassifier(loss="log")
        file_name = model.__class__.__name__ + '_TfIdfVec_.pmml'
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', model)
        ])
        pipeline.fit(X, Y)
        skl_to_pmml(pipeline, features , target, file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_sgd_with_text_cv(self):
        categories = ['alt.atheism','talk.religion.misc']
        data = fetch_20newsgroups(subset='train', categories=categories)
        X = data.data[:4]
        Y = data.target[:4]
        features = ['input']
        target = 'output'
        model = SGDClassifier(loss="log")
        file_name = model.__class__.__name__ + '_CountVec_.pmml'
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', model)
        ])
        pipeline.fit(X, Y)
        skl_to_pmml(pipeline, features , target, file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_isolation_forest(self):
        iris = datasets.load_iris()
        X = iris.data
        features = iris.feature_names
        model = IsolationForest()
        pipeline = Pipeline([
            ('standard_scaler',StandardScaler()),
            ('Imputer',Imputer()),
            ('model',model)
        ])
        pipeline.fit(X)
        file_name = model.__class__.__name__+'.pmml'
        skl_to_pmml(pipeline, features ,pmml_f_name= file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_ocsvm(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        features = iris.feature_names
        model = OneClassSVM()
        pipeline = Pipeline([
            ('standard_scaler',StandardScaler()),
            ('Imputer',Imputer()),
            ('model',model)
        ])
        pipeline.fit(X,y)
        file_name = model.__class__.__name__+'.pmml'
        skl_to_pmml(pipeline, features ,pmml_f_name= file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_lag(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        features = iris.feature_names
        model = LogisticRegression()
        pipeline = Pipeline([
            ('lag',Lag(aggregation="stddev", value=3)),
            ('model',model)
        ])
        pipeline.fit(X,y)
        file_name = model.__class__.__name__+'lag_stddev.pmml'
        skl_to_pmml(pipeline, features , 'species',pmml_f_name= file_name)
        self.assertEqual(self.schema.is_valid(file_name), True)

    
    #Exponential Smoothing Test cases
    def test_exponentialSmoothing_01(self):
        ts_data = self.statsmodels_data_helper.get_data_with_trend_and_seasonality()        
        f_name='exponential_smoothing1.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=True, 
                                        seasonal='add', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)


    #Non Seasonal Arima Test cases
    def test_non_seasonal_arima1(self):
        ts_data = self.statsmodels_data_helper.get_non_seasonal_data()
        f_name='non_seasonal_arima1.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle')
        StatsmodelsToPmml(result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima2(self):
        ts_data = self.statsmodels_data_helper.get_non_seasonal_data()
        f_name='non_seasonal_arima1.pmml'
        model = StateSpaceARIMA(ts_data,order=(3, 1, 2),trend='c')
        result = model.fit()
        StatsmodelsToPmml(result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    @unittest.skip("")
    def test_non_seasonal_arima7(self):
        ts_data = self.statsmodels_data_helper.get_non_seasonal_data()
        f_name='non_seasonal_arima7.pmml'
        model = ARIMA(ts_data,order=(5, 1, 2))
        result = model.fit(trend = 'nc', method = 'mle')
        StatsmodelsToPmml(result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    @unittest.skip("")
    def test_non_seasonal_arima8(self):
        ts_data = self.statsmodels_data_helper.get_non_seasonal_data()
        f_name='non_seasonal_arima8.pmml'
        model = ARIMA(ts_data,order=(5, 1, 2))
        result = model.fit(trend = 'c', method = 'mle')
        StatsmodelsToPmml(result, f_name,conf_int=[80,95])
        self.assertEqual(self.schema.is_valid(f_name),True)


    #Seasonal Arima Test cases
    def test_seasonal_arima1(self):
        ts_data = self.statsmodels_data_helper.get_seasonal_data()
        f_name='seasonal_arima1.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (0, 0, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 'c')
        result = model.fit()
        StatsmodelsToPmml(result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima2(self):
        ts_data = self.statsmodels_data_helper.get_seasonal_data()
        f_name='seasonal_arima2.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12))
        result = model.fit()
        StatsmodelsToPmml(result, f_name, conf_int=[80])
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_varmax_with_intercept(self):
        ts_data = self.statsmodels_data_helper.get_data_for_varmax()
        f_name='varmax_with_intercept.pmml'
        model = VARMAX(ts_data, order=(1,1))
        result = model.fit()
        StatsmodelsToPmml(result, f_name, conf_int=[80,95])
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_varmax_without_intercept(self):
        ts_data = self.statsmodels_data_helper.get_data_for_varmax()
        f_name='varmax_without_intercept.pmml'
        model = VARMAX(ts_data, order=(1,1), trend=None)
        result = model.fit()
        StatsmodelsToPmml(result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)


if __name__ == "__main__":
    unittest.main(warnings='ignore')