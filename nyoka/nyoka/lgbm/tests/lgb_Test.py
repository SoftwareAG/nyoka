import unittest
from lightgbm import LGBMRegressor,LGBMClassifier
from nyoka.nyoka import PMML43Ext as pml
import nyoka.nyoka.lgbm.lgb_to_pmml as lgbToPmml
import pandas as pd
from sklearn.model_selection import train_test_split

class TestMethods(unittest.TestCase):

    def test_LightGradientBoostingClassifier(self):
        model=LGBMClassifier()
        sk_model,feature_names,target_name=iris_dataset(model)
        derived_col_names=feature_names
        mining_imp_value=()
        categoric_values=()

        self.assertEqual(
            lgbToPmml.get_segments_for_lgbc(sk_model,derived_col_names,feature_names,target_name,mining_imp_value,
                                                         categoric_values)[0].__class__.__name__,
                        pml.Segment().__class__.__name__ )

        self.assertEqual(
            lgbToPmml.get_segments_for_lgbc(sk_model, derived_col_names, feature_names, target_name, mining_imp_value,
                                            categoric_values)[-1].get_RegressionModel().__class__.__name__,
            pml.RegressionModel().__class__.__name__)

        self.assertEqual(
            len(lgbToPmml.get_segments_for_lgbc(sk_model, derived_col_names, feature_names, target_name, mining_imp_value,
                                            categoric_values)),
            sk_model.n_classes_+1)

        self.assertEqual(
            lgbToPmml.get_ensemble_models(sk_model, derived_col_names, feature_names, target_name,
                                          mining_imp_value,
                                          categoric_values)[0].get_functionName(),
            'classification')

        self.assertEqual(
            len(lgbToPmml.get_ensemble_models(sk_model, derived_col_names, feature_names, target_name,
                                              mining_imp_value,
                                              categoric_values)[0].get_MiningSchema().get_MiningField()),
            model.n_features_ + 1)

    def test_LightGradientBoostingRegressor(self):
        model=LGBMRegressor()
        sk_model, feature_names, target_name = auto_dataset_for_regression(model)
        derived_col_names = feature_names
        mining_imp_value = ()
        categoric_values = ()

        self.assertEqual(
            lgbToPmml.get_segments_for_lgbr(sk_model, derived_col_names, feature_names, target_name, mining_imp_value,
                                            categoric_values).__class__.__name__,
            pml.Segmentation().__class__.__name__)

        self.assertEqual(
            len(lgbToPmml.get_segments_for_lgbr(sk_model, derived_col_names, feature_names, target_name, mining_imp_value,
                                            categoric_values).get_Segment()),
            model.n_estimators)

        self.assertEqual(
            lgbToPmml.get_ensemble_models(sk_model, derived_col_names, feature_names, target_name,
                                                mining_imp_value,
                                                categoric_values)[0].__class__.__name__,
            pml.MiningModel().__class__.__name__)

        self.assertEqual(
            lgbToPmml.get_ensemble_models(sk_model, derived_col_names, feature_names, target_name,
                                          mining_imp_value,
                                          categoric_values)[0].get_functionName(),
            'regression')

        self.assertEqual(
            len(lgbToPmml.get_ensemble_models(sk_model, derived_col_names, feature_names, target_name,
                                          mining_imp_value,
                                          categoric_values)[0].get_MiningSchema().get_MiningField()),
            model.n_features_+1)






def iris_dataset(sk_model):
    df = pd.read_csv('iris.csv')
    feature_names = df.columns.drop('species')
    target_name = 'species'
    x_train, x_test, y_train, y_test = train_test_split(df[feature_names], df[target_name], test_size=0.33,
                                                        random_state=101)
    sk_model=sk_model.fit(x_train,y_train)
    return sk_model,feature_names,target_name

def auto_dataset_for_regression(sk_model):
    df = pd.read_csv('auto-mpg.csv')
    X = df.drop(['mpg','car name'], axis=1)
    y = df['mpg']
    feature_names = [name for name in df.columns if name not in ('mpg','car name')]
    target_name='mpg'
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=101)
    sk_model=sk_model.fit(x_train,y_train)
    return sk_model,feature_names,target_name

if __name__=='__main__':
    unittest.main(warnings='ignore')