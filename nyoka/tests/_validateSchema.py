# Exporters
from nyoka import skl_to_pmml, KerasToPmml, ArimaToPMML, ExponentialSmoothingToPMML

# Pipeline/ DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

# Sklearn Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, LabelEncoder, Imputer,\
 Binarizer, PolynomialFeatures, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

## Sklearn models
# Linear models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

# Tree models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# SVM
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

# Ensemble
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor

# Clustering
from sklearn.cluster import KMeans

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# keras models
from keras.applications import MobileNet, ResNet50, VGG16, Xception, InceptionV3, DenseNet121
from keras.layers import Input

# statsmodels models
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import unittest
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
import xmlschema
import pandas as pd

class StatsmodelsDataHelper:
    def getData1(self):
        # data with trend and seasonality present
        # no of international visitors in Australia
        data = [41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015, 31.1802, 37.7179,
                40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668, 35.1919,
                44.3197, 47.9137]
        index = pd.DatetimeIndex(start='2005', end='2010-Q4', freq='QS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_visitors'
        return ts_data
		
    def getData2(self):
		# data with trend but no seasonality
        # no. of annual passengers of air carriers registered in Australia
        data = [17.5534, 21.86, 23.8866, 26.9293, 26.8885, 28.8314, 30.0751, 30.9535, 30.1857, 31.5797, 32.5776,
                33.4774, 39.0216, 41.3864, 41.5966]
        index = pd.DatetimeIndex(start='1990', end='2005', freq='A')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_passengers'
        return ts_data

    def getData3(self):
		# data with no trend and no seasonality
        # Oil production in Saudi Arabia
        data = [446.6565, 454.4733, 455.663, 423.6322, 456.2713, 440.5881, 425.3325, 485.1494, 506.0482, 526.792,
                514.2689, 494.211]
        index = pd.DatetimeIndex(start='1996', end='2008', freq='A')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'oil_production'
        return ts_data

    def getData4(self):
		# Non Seasonal Data
        data = [266,146,183,119,180,169,232,225,193,123,337,186,194,150,210,273,191,287,
                226,304,290,422,265,342,340,440,316,439,401,390,490,408,490,420,520,480]
        index = pd.DatetimeIndex(start='2016-01-01', end='2018-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'date_index'
        ts_data.name = 'cars_sold'
        return ts_data

    def getData5(self):
		# Seasonal Data
        data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150,
                178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235,
                229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315,
                364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467,
                404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407,
                362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432]
        index = pd.DatetimeIndex(start='1949-01-01', end='1960-12-01', freq='MS')
        ts_data = pd.Series(data, index)
        ts_data.index.name = 'datetime_index'
        ts_data.name = 'n_passengers'
        return ts_data
    

class PmmlValidation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Starting pmml validation tests.")
        data = datasets.load_iris()
        self.X = data.data
        self.y = data.target
        self.y_bin = [i%2 for i in range(self.X.shape[0])]
        self.features = data.feature_names
        data = datasets.load_boston()
        self.X_reg = data.data
        self.y_reg = data.target
        self.features_reg =  data.feature_names
        self.schema = xmlschema.XMLSchema("nyoka/pmml44New.xsd")
        self.statsmodels_data_helper = StatsmodelsDataHelper()

    
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

    def test_validate_keras_mobilenet(self):
        input_tensor = Input(shape=(224, 224, 3))
        model = MobileNet(weights="imagenet", input_tensor=input_tensor)
        file_name = "keras"+model.name+".pmml"
        pmml_obj = KerasToPmml(model,dataSet="image",predictedClasses=[str(i) for i in range(1000)])
        pmml_obj.export(open(file_name,'w'),0)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_keras_resnet(self):
        input_tensor = Input(shape=(224, 224, 3))
        model = ResNet50(weights="imagenet", input_tensor=input_tensor)
        file_name = "keras"+model.name+".pmml"
        pmml_obj = KerasToPmml(model,dataSet="image",predictedClasses=[str(i) for i in range(1000)])
        pmml_obj.export(open(file_name,'w'),0)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_keras_vgg(self):
        input_tensor = Input(shape=(224, 224, 3))
        model = VGG16(weights="imagenet", input_tensor=input_tensor)
        file_name = "keras"+model.name+".pmml"
        pmml_obj = KerasToPmml(model,dataSet="image",predictedClasses=[str(i) for i in range(1000)])
        pmml_obj.export(open(file_name,'w'),0)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_keras_inception(self):
        input_tensor = Input(shape=(224, 224, 3))
        model = InceptionV3(weights="imagenet", input_tensor=input_tensor)
        file_name = "keras"+model.name+".pmml"
        pmml_obj = KerasToPmml(model,dataSet="image",predictedClasses=[str(i) for i in range(1000)])
        pmml_obj.export(open(file_name,'w'),0)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_keras_xception(self):
        input_tensor = Input(shape=(224, 224, 3))
        model = Xception(weights="imagenet", input_tensor=input_tensor)
        file_name = "keras"+model.name+".pmml"
        pmml_obj = KerasToPmml(model,dataSet="image",predictedClasses=[str(i) for i in range(1000)])
        pmml_obj.export(open(file_name,'w'),0)
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_keras_densenet(self):
        input_tensor = Input(shape=(224, 224, 3))
        model = DenseNet121(weights="imagenet", input_tensor=input_tensor)
        file_name = "keras"+model.name+".pmml"
        pmml_obj = KerasToPmml(model,dataSet="image",predictedClasses=[str(i) for i in range(1000)])
        pmml_obj.export(open(file_name,'w'),0)
        self.assertEqual(self.schema.is_valid(file_name), True)

    
    #Exponential Smoothing Test cases
    def test_exponentialSmoothing_01(self):
        ts_data = self.statsmodels_data_helper.getData1()        
        f_name='exponential_smoothing1.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=True, 
                                        seasonal='add', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)
        
    def test_exponentialSmoothing_02(self):
        ts_data = self.statsmodels_data_helper.getData1()        
        f_name='exponential_smoothing2.pmml'        
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=False, 
                                        seasonal='add', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_03(self):
        ts_data = self.statsmodels_data_helper.getData1()        
        f_name='exponential_smoothing3.pmml'                
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=True, 
                                        seasonal='mul', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_04(self):
        ts_data = self.statsmodels_data_helper.getData1()       
        f_name='exponential_smoothing4.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=False, 
                                        seasonal='mul', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_05(self):
        ts_data = self.statsmodels_data_helper.getData1()        
        f_name='exponential_smoothing5.pmml'        
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=True, 
                                        seasonal='add', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_06(self):
        ts_data = self.statsmodels_data_helper.getData1()        
        f_name='exponential_smoothing6.pmml'                
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=False, 
                                        seasonal='add', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_07(self):
        ts_data = self.statsmodels_data_helper.getData1()        
        f_name='exponential_smoothing7.pmml'                
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=True, 
                                        seasonal='mul', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_08(self):
        ts_data = self.statsmodels_data_helper.getData1()        
        f_name='exponential_smoothing8.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=False, 
                                        seasonal='mul', 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)
        
    def test_exponentialSmoothing_09(self):
        ts_data = self.statsmodels_data_helper.getData2()        
        f_name='exponential_smoothing9.pmml'        
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=True, 
                                        seasonal=None, 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_10(self):
        ts_data = self.statsmodels_data_helper.getData2()       
        f_name='exponential_smoothing10.pmml'               
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=True, 
                                        seasonal=None, 
                                        seasonal_periods=None)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_11(self):
        ts_data = self.statsmodels_data_helper.getData2()       
        f_name='exponential_smoothing11.pmml'                
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=False, 
                                        seasonal=None, 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_12(self):
        ts_data = self.statsmodels_data_helper.getData2()
        f_name='exponential_smoothing12.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='add', 
                                        damped=False, 
                                        seasonal=None, 
                                        seasonal_periods=None)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_13(self):
        ts_data = self.statsmodels_data_helper.getData2()       
        f_name='exponential_smoothing13.pmml'                
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=True, 
                                        seasonal=None, 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_14(self):
        ts_data = self.statsmodels_data_helper.getData2()        
        f_name='exponential_smoothing14.pmml'                
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=True, 
                                        seasonal=None, 
                                        seasonal_periods=None)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_15(self):
        ts_data = self.statsmodels_data_helper.getData2()  
        f_name='exponential_smoothing15.pmml'        
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=False, 
                                        seasonal=None, 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_exponentialSmoothing_16(self):
        ts_data = self.statsmodels_data_helper.getData2()
        f_name='exponential_smoothing16.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend='mul', 
                                        damped=False, 
                                        seasonal=None, 
                                        seasonal_periods=None)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)
        
    def test_exponentialSmoothing_17(self):
        ts_data = self.statsmodels_data_helper.getData3()
        f_name='exponential_smoothing17.pmml'
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend=None, 
                                        damped=False, 
                                        seasonal=None, 
                                        seasonal_periods=None)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)
        
    def test_exponentialSmoothing_18(self):
        ts_data = self.statsmodels_data_helper.getData3()
        f_name='exponential_smoothing18.pmml'
        
        model_obj = ExponentialSmoothing(ts_data, 
                                        trend=None, 
                                        damped=False, 
                                        seasonal=None, 
                                        seasonal_periods=2)
        results_obj = model_obj.fit(optimized=True)
        
        ExponentialSmoothingToPMML(ts_data, model_obj,results_obj, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)


    #Non Seasonal Arima Test cases
    def test_non_seasonal_arima1(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima1.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle', solver = 'lbfgs')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima2(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima2.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle', solver = 'nm')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima3(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima3.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle', solver = 'bfgs')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima4(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima4.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle', solver = 'powell')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima5(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima5.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle', solver = 'cg')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima6(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima6.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css-mle', solver = 'ncg')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima7(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima7.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'mle', solver = 'lbfgs')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima8(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima8.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'mle', solver = 'nm')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima9(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima9.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'mle', solver = 'bfgs')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima10(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima10.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'mle', solver = 'powell')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima11(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima11.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'mle', solver = 'cg')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima12(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima12.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'mle', solver = 'ncg')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima13(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima13.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css', solver = 'lbfgs')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima14(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima14.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css', solver = 'nm')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima15(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima15.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css', solver = 'bfgs')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima16(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima16.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css', solver = 'powell')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima17(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima17.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css', solver = 'cg')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_non_seasonal_arima18(self):
        ts_data = self.statsmodels_data_helper.getData4()
        f_name='non_seasonal_arima18.pmml'
        model = ARIMA(ts_data,order=(9, 2, 0))
        result = model.fit(trend = 'c', method = 'css', solver = 'ncg')
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)


    #Seasonal Arima Test cases
    def test_seasonal_arima1(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima1.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = True, 
                                        time_varying_regression = True, 
                                        mle_regression = False, 
                                        simple_differencing = True, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = True, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima2(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima2.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = True, 
                                        time_varying_regression = True, 
                                        mle_regression = False, 
                                        simple_differencing = False, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = False, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima3(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima3.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = True, 
                                        time_varying_regression = False, 
                                        mle_regression = True, 
                                        simple_differencing = True, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = True, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima4(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima4.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = True, 
                                        time_varying_regression = False, 
                                        mle_regression = True, 
                                        simple_differencing = False, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = False, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima5(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima5.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = False, 
                                        time_varying_regression = True, 
                                        mle_regression = False, 
                                        simple_differencing = True, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = True, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima6(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima6.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = False, 
                                        time_varying_regression = True, 
                                        mle_regression = False, 
                                        simple_differencing = False, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = False, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima7(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima7.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = False, 
                                        time_varying_regression = False, 
                                        mle_regression = True, 
                                        simple_differencing = True, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = True, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

    def test_seasonal_arima8(self):
        ts_data = self.statsmodels_data_helper.getData5()
        f_name='seasonal_arima8.pmml'
        model = SARIMAX(endog = ts_data,
                                        exog = None,
                                        order = (3, 1, 1),
                                        seasonal_order = (3, 1, 1, 12),
                                        trend = 't',
                                        measurement_error = False, 
                                        time_varying_regression = False, 
                                        mle_regression = True, 
                                        simple_differencing = False, 
                                        enforce_stationarity = False, 
                                        enforce_invertibility = False, 
                                        hamilton_representation = False, 
                                        concentrate_scale = False)
        result = model.fit()
        ArimaToPMML(ts_data, model, result, f_name)
        self.assertEqual(self.schema.is_valid(f_name),True)

if __name__ == "__main__":
    unittest.main(warnings='ignore')