import os
import unittest
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from nyoka import model_to_pmml
from nyoka.reconstruct import pmml_to_lgbmTrainAPI
from nyoka.PMML43Ext import parse


class TestMethods(unittest.TestCase):
    
    def test_lgbm_export_and_reconstruct_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target
        features = irisd.columns.drop('Species')
        target = 'Species'

        X_train, X_test, y_train, y_test = train_test_split(irisd[features], irisd[target], test_size=0.3, random_state=0) 

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train , free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class':3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 20
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=50,
                        verbose_eval = False
                    )

        originalPredictions = gbm.predict(X_test)
        
        pmml_file_name = "LGBM_Train_API_Iris.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':params,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':gbm,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'trainAndscore'
            }
        }

        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        #Reconstruction
        pmml_obj = parse(pmml_file_name, silence=True)
        reconstructed_gbm = pmml_to_lgbmTrainAPI.reconstruct(pmml_obj)
        rcModelPredictions = reconstructed_gbm.predict(X_test)

        self.assertEqual(originalPredictions.all() == rcModelPredictions.all(),True)     

if __name__=='__main__':
    unittest.main(warnings='ignore')







