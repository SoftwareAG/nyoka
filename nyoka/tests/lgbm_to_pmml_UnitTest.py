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
from nyoka import model_to_pmml


class TestMethods(unittest.TestCase):

    
    def test_lgbm_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        lgbmc = LGBMClassifier()
        lgbmc.fit(irisd[features],irisd[target])

        pmml_file_name = "lgbmc_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':lgbmc,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_lgbm_02(self):

        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg','car name'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in ('mpg','car name')]
        target_name='mpg'

        lgbmr = LGBMRegressor()
        lgbmr.fit(auto[feature_names],auto[target_name])

        pmml_file_name = "lgbmr_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':lgbmr,
                'featuresUsed':feature_names,
                'targetName':target_name,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_lgbm_03(self):
        
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('scaling',StandardScaler())
        ])

        X = pipeline_obj.fit_transform(irisd[features])

        lgbmc = LGBMClassifier(n_estimators=5)
        lgbmc.fit(X,irisd[target])

        pmml_file_name = "lgbmc_pmml_preprocess.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':lgbmc,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)

    def test_lgbm_04(self):
        
        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in ('mpg')]

        target_name = 'mpg'
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', CountVectorizer()),
                (['displacement'],[StandardScaler()]) 
            ]))
        ])
        X = pipeline_obj.fit_transform(x_train)

        lgbmr = LGBMRegressor()
        lgbmr.fit(X,y_train)

        pmml_file_name = "lgbmr_pmml_preprocess.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':lgbmr,
                'featuresUsed':feature_names,
                'targetName':target_name,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)

if __name__=='__main__':
    unittest.main(warnings='ignore')