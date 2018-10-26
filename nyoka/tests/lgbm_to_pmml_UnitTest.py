import os
import unittest
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor,LGBMClassifier
from nyoka import lgb_to_pmml


class TestMethods(unittest.TestCase):

    
    def test_lgbm_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('lgbmc',LGBMClassifier())
        ])

        pipeline_obj.fit(irisd[features],irisd[target])

        lgb_to_pmml(pipeline_obj,features,target,"lgbmc_pmml.pmml")

        self.assertEqual(os.path.isfile("lgbmc_pmml.pmml"),True)


    def test_lgbm_02(self):

        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg','car name'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in ('mpg','car name')]
        target_name='mpg'

        pipeline_obj = Pipeline([
            ('lgbmr',LGBMRegressor())
        ])

        pipeline_obj.fit(auto[feature_names],auto[target_name])

        lgb_to_pmml(pipeline_obj,feature_names,target_name,"lgbmr_pmml.pmml")

        self.assertEqual(os.path.isfile("lgbmr_pmml.pmml"),True)


    def test_lgbm_03(self):
        
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('scaling',StandardScaler()), 
            ('LGBMC_preprocess',LGBMClassifier(n_estimators=5))
        ])

        pipeline_obj.fit(irisd[features],irisd[target])

        lgb_to_pmml(pipeline_obj,features,target,"lgbmc_pmml_preprocess.pmml")

        self.assertEqual(os.path.isfile("lgbmc_pmml_preprocess.pmml"),True)

    def test_lgbm_04(self):
        
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
            ('lgbmr',LGBMRegressor())
        ])
        pipeline_obj.fit(x_train,y_train)
        
        lgb_to_pmml(pipeline_obj,feature_names,target_name,"lgbmr_pmml_preprocess.pmml")

        self.assertEqual(os.path.isfile("lgbmr_pmml_preprocess.pmml"),True)

if __name__=='__main__':
    unittest.main(warnings='ignore')







