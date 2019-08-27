import os
import unittest
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor,XGBClassifier
from nyoka import xgboost_to_pmml


class TestMethods(unittest.TestCase):

    
    def test_xgboost_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('lgbmc',XGBClassifier())
        ])

        pipeline_obj.fit(irisd[features],irisd[target])

        xgboost_to_pmml(pipeline_obj,features,target,"xgbc_pmml.pmml")

        self.assertEqual(os.path.isfile("xgbc_pmml.pmml"),True)


    def test_xgboost_02(self):

        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg','car name'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in ('mpg','car name')]
        target_name='mpg'

        pipeline_obj = Pipeline([
            ('lgbmr',XGBRegressor())
        ])

        pipeline_obj.fit(auto[feature_names],auto[target_name])

        xgboost_to_pmml(pipeline_obj,feature_names,target_name,"xgbr_pmml.pmml")

        self.assertEqual(os.path.isfile("xgbr_pmml.pmml"),True)


    def test_xgboost_03(self):
        
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('scaling',StandardScaler()), 
            ('LGBMC_preprocess',XGBClassifier(n_estimators=5))
        ])

        pipeline_obj.fit(irisd[features],irisd[target])

        xgboost_to_pmml(pipeline_obj,features,target,"xgbc_pmml_preprocess.pmml")

        self.assertEqual(os.path.isfile("xgbc_pmml_preprocess.pmml"),True)

    def test_xgboost_04(self):
        
        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in ('mpg')]

        target_name='mpg'
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', CountVectorizer()),
                (['displacement'],[StandardScaler()]) 
            ])),
            ('lgbmr',XGBRegressor())
        ])
        pipeline_obj.fit(x_train,y_train)
        
        xgboost_to_pmml(pipeline_obj,feature_names,target_name,"xgbr_pmml_preprocess.pmml")

        self.assertEqual(os.path.isfile("xgbr_pmml_preprocess.pmml"),True)

if __name__=='__main__':
    unittest.main(warnings='ignore')







