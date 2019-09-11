import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, LabelBinarizer,\
     Binarizer, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn_pandas import DataFrameMapper
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from nyoka import skl_to_pmml
from nyoka import PMML44 as pml
import unittest
import ast
import numpy
from adapaUtilities import AdapaUtility
from dataUtilities import DataUtility

class TestCases(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("******* Unit Test for sklearn *******")
        self.data_utility = DataUtility()
        self.adapa_utility = AdapaUtility()

    def test_01_linear_regression(self):
        print("\ntest 01 (linear regression without preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = LinearRegression()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test01sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    def test_02_linear_regression_with_scaler(self):
        print("\ntest 02 (linear regression with preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = LinearRegression()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test02sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    def test_03_logistic_regression_with_scaler(self):
        print("\ntest 03 (logistic regression with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("mapper", DataFrameMapper([
                    (["sepal length (cm)", "sepal width (cm)"], MinMaxScaler()),
                    (["petal length (cm)", "petal width (cm)"], None)
                ])
            ),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test03sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_04_logistic_regression_with_scaler(self):
        print("\ntest 04 (logistic regression with preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test04sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_05_logistic_regression(self):
        print("\ntest 05 (logistic regression without preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test05sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_06_logistic_regression(self):
        print("\ntest 06 (logistic regression without preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test06sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_07_ridge_classifier(self):
        print("\ntest 07 (Ridge Classifier) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = RidgeClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test07sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = model._predict_proba_lr(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_08_ridge_classifier(self):
        print("\ntest 08 (Ridge Classifier) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = RidgeClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test08sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = model._predict_proba_lr(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    @unittest.skip("")
    def test_09_sgd_classifier(self):
        print("\ntest 09 (SGD Classifier with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = SGDClassifier(loss="log")
        pipeline_obj = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test09sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_10_sgd_classifier(self):
        print("\ntest 10 (SGD Classifier with preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = SGDClassifier(loss="log")
        pipeline_obj = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test10sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_11_lda(self):
        print("\ntest 11 (LDA with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = LinearDiscriminantAnalysis()
        pipeline_obj = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test11sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_12_lda(self):
        print("\ntest 12 (LDA with preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = LinearDiscriminantAnalysis()
        pipeline_obj = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test12sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_13_linearsvc(self):
        print("\ntest 13 (LinearSVC with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = LinearSVC()
        pipeline_obj = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test13sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.decision_function(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_14_linearsvc(self):
        print("\ntest 14 (LinearSVC with preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = LinearSVC()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test14sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = model._predict_proba_lr(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_15_linearsvr(self):
        print("\ntest 15 (linear svr without preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = LinearSVR()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test15sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    def test_16_linearsvr(self):
        print("\ntest 16 (linear svr with preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = LinearSVR()
        pipeline_obj = Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test16sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    def test_17_decisiontreeclassifier(self):
        print("\ntest 17 (decision tree classifier with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = DecisionTreeClassifier()
        pipeline_obj = Pipeline([
            ("scaler", Binarizer()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test17sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_18_decisiontreeclassifier(self):
        print("\ntest 18 (decision tree classifier with preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = DecisionTreeClassifier()
        pipeline_obj = Pipeline([
            ("scaler", Binarizer()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test18sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_19_decisiontreeclassifier(self):
        print("\ntest 19 (decision tree classifier without preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = DecisionTreeClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test19sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_20_decisiontreeclassifier(self):
        print("\ntest 20 (decision tree classifier without preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = DecisionTreeClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test20sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_21_svr(self):
        print("\ntest 21 (SVR without preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = SVR()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test21sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    def test_22_gaussian_nb(self):
        print("\ntest 22 (GaussianNB without preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = GaussianNB()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test22sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_23_gaussian_nb(self):
        print("\ntest 23 (GaussianNB without preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = GaussianNB()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test23sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    def test_24_gaussian_nb(self):
        print("\ntest 24 (GaussianNB with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = GaussianNB()
        pipeline_obj = Pipeline([
            ('scaler', StandardScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test24sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    @unittest.skip("")
    def test_25_random_forest_regressor(self):
        print("\ntest 25 (random forest regressor without preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = RandomForestRegressor()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test25sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    @unittest.skip("")
    def test_26_random_forest_classifier(self):
        print("\ntest 26 (random forest classifier with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = RandomForestClassifier()
        pipeline_obj = Pipeline([
            ('scaler',MinMaxScaler()),
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test26sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    @unittest.skip("")
    def test_27_random_forest_classifier(self):
        print("\ntest 27 (random forest classifier with preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = RandomForestClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test27sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)


    def test_28_gradient_boosting_classifier(self):
        print("\ntest 28 (gradient boosting classifier with preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = GradientBoostingClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test28sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    @unittest.skip("")
    def test_29_gradient_boosting_classifier(self):
        print("\ntest 29 (gradient boosting classifier with preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = GradientBoostingClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test29sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)


    def test_30_gradient_boosting_regressor(self):
        print("\ntest 30 (gradient boosting regressor without preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = GradientBoostingRegressor()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test30sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    @unittest.skip("")
    def test_31_knn_classifier(self):
        print("\ntest 31 (knn classifier without preprocessing) [binary-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_binary_classification()

        model = KNeighborsClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test31sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)


    def test_32_knn_classifier(self):
        print("\ntest 32 (knn classifier without preprocessing) [multi-class]\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = KNeighborsClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test32sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.predict_proba(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)


    def test_33_knn_regressor(self):
        print("\ntest 33 (knn regressor without preprocessing)\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_regression()

        model = KNeighborsRegressor()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test33sklearn.pmml'
        
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, _ = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)

    def test_34_kmeans(self):
        print("\ntest 34 (kmeans without preprocessing\n")
        X, X_test, y, features, target, test_file = self.data_utility.get_data_for_multi_class_classification()

        model = KMeans(n_clusters=2)
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X,y)
        file_name = 'test34sklearn.pmml'
        skl_to_pmml(pipeline_obj, features, target, file_name)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file)
        model_pred = pipeline_obj.predict(X_test)
        model_prob = pipeline_obj.transform(X_test)
        self.assertEqual(self.adapa_utility.compare_predictions(predictions, model_pred), True)
        self.assertEqual(self.adapa_utility.compare_probability(probabilities, model_prob), True)

    @unittest.skip("")
    def test_35_isolation_forest(self):
        print("\ntest 34 (Isolation Forest\n")
        detection_map = {
            'true': -1,
            'false': 1
        }
        X = numpy.array([
            [1,2,3,4],
            [2,1,3,4],
            [3,2,1,4],
            [3,2,4,1],
            [4,3,2,1],
            [2,4,3,1]
        ], dtype=numpy.float32)
        test_data = numpy.array([[0,4,0,7],[4,0,4,7]])
        features = ['a','b','c','d']
        model = IsolationForest(n_estimators=40,contamination=0)
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(X)
        file_name = 'test35sklearn.pmml'
        skl_to_pmml(pipeline_obj, features, '', file_name)
        model_pred = pipeline_obj.predict(test_data)
        model_scores = model.score_samples(test_data)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        z_predictions = self.adapa_utility.score_in_zserver(model_name,'nyoka/tests/test_forest.csv','ANOMALY')
        cnt = 0
        for idx, value in enumerate(z_predictions):
            score, is_anomaly = value.split(",")
            score = -1 * float(score)
            if "{:.6f}".format(score) != "{:.6f}".format(model_scores[idx]) or model_pred[idx] != detection_map[is_anomaly]:
                cnt += 1
        self.assertEqual(cnt,0)

    @unittest.skip("")
    def test_36_one_class_svm(self):
        print("\ntest 36 (One Class SVM\n")
        detection_map = {
            'true': -1,
            'false': 1
        }
        df = pd.read_csv("nyoka/tests/train_ocsvm.csv")
        df_test = pd.read_csv("nyoka/tests/test_ocsvm.csv")
        features = df.columns
        model = OneClassSVM(nu=0.1)
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(df)
        file_name = 'test36sklearn.pmml'
        skl_to_pmml(pipeline_obj, features, '', file_name)
        model_pred = pipeline_obj.predict(df_test)
        model_scores = pipeline_obj.decision_function(df_test)
        model_name  = self.adapa_utility.upload_to_zserver(file_name)
        z_predictions = self.adapa_utility.score_in_zserver(model_name,'nyoka/tests/test_ocsvm.csv','ANOMALY')
        cnt = 0
        for idx, value in enumerate(z_predictions):
            score, is_anomaly = value.split(",")
            score = float(score)
            if "{:.6f}".format(score) != "{:.6f}".format(model_scores[idx]) or model_pred[idx] != detection_map[is_anomaly]:
                cnt += 1
        self.assertEqual(cnt,0)


    @classmethod
    def tearDownClass(self):
        print("\n******* Finished *******\n")
            
if __name__ == '__main__':
    unittest.main(warnings='ignore')