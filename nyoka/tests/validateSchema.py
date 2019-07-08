# Exporters
from nyoka import skl_to_pmml, xgboost_to_pmml, lgb_to_pmml, KerasToPmml, ArimaToPMML, ExponentialSmoothingToPMML

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer, Binarizer, PolynomialFeatures, LabelBinarizer

# Sklearn models
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,\
     ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# xgboost models
from xgboost import XGBClassifier, XGBRegressor

# lgbm models
from lightgbm import LGBMClassifier, LGBMRegressor

# keras models
from keras.applications import MobileNet, ResNet50

import unittest
from sklearn import datasets
import xmlschema

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
        self.schema = xmlschema.XMLSchema("../pmml44.xsd")

    
    def test_validate_sklearn_linear_models_binary_class(self):
        model = LogisticRegression()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y_bin)
        file_name = 'linear_model_binary_classification.pmml'
        skl_to_pmml(pipe, self.features, 'binary',file_name)
        print(self.schema.validate(file_name))
        self.assertEqual(self.schema.is_valid(file_name), True)
   
    def test_validate_sklearn_linear_models_multiclass(self):
        model = LogisticRegression()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X, self.y)
        file_name = 'linear_model_multi_class_classification.pmml'
        skl_to_pmml(pipe, self.features, 'species',file_name)
        print(self.schema.validate(file_name))
        self.assertEqual(self.schema.is_valid(file_name), True)

    def test_validate_sklearn_linear_models_regression(self):
        model = LinearRegression()
        pipe = Pipeline([
            ('model',model)
        ])
        pipe.fit(self.X_reg, self.y_reg)
        file_name = 'linear_model_regression.pmml'
        skl_to_pmml(pipe, self.features_reg, 'target',file_name)
        print(self.schema.validate(file_name))
        self.assertEqual(self.schema.is_valid(file_name), True)


if __name__ == "__main__":
    unittest.main(warnings='ignore')