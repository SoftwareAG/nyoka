import unittest
from xgboost import XGBRegressor,XGBClassifier
from nyoka.nyoka import PMML43Ext as pml
import nyoka.nyoka.xgboost.xgboost_to_pmml as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

class TestMethods(unittest.TestCase):

    def test_XGBoost(self):
        model=XGBClassifier()
        sk_model=iris_dataset(model)
        derived_col_names=['sepal_length','petal_length']
        col_names=['sepal_length','petal_length']
        target_name='species'
        mining_imp_value=()
        categoric_values=()

        self.assertEqual(
            len(xgb.get_segments_for_xgbc(sk_model,derived_col_names,col_names,target_name,mining_imp_value,categoric_values)),
            4)
        self.assertEqual(
            xgb.get_segments_for_xgbc(sk_model,derived_col_names,col_names,target_name,mining_imp_value,categoric_values)[
                3].get_id(),4)
        self.assertEqual(
            xgb.get_segments_for_xgbc(sk_model, derived_col_names, col_names, target_name, mining_imp_value,categoric_values)[
                3].get_RegressionModel().get_RegressionTable()[0].get_intercept(), '0.0')



        self.assertEqual(
            xgb.mining_Field_For_First_Segment(col_names).__class__.__name__,
            pml.MiningSchema().__class__.__name__)
        self.assertEqual(
            xgb.mining_Field_For_First_Segment(col_names).get_MiningField()[0].get_name(),
            'sepal_length')
        self.assertEqual(len(xgb.mining_Field_For_First_Segment(['a','b','d','e']).get_MiningField())
                         ,4)


        self.assertEqual(
            type(xgb.generate_Segments_Equal_To_Estimators([],derived_col_names,col_names)),
            type([]))


        self.assertEqual(
            xgb.add_segmentation(sk_model,[],[],pml.Output,1).__class__.__name__,
            pml.Segment().__class__.__name__)
        self.assertEqual(
            xgb.add_segmentation(sk_model,[],[],pml.Output,1).get_MiningModel().__class__.__name__,
            pml.MiningModel().__class__.__name__)
        self.assertEqual(xgb.add_segmentation(sk_model, [], [], pml.Output, 1).get_id(),
                         2)

        self.assertEqual(type(xgb.get_regrs_tabl(sk_model,col_names,'species',categoric_values)),type([]))



def iris_dataset(sk_model):
    df = pd.read_csv('iris.csv')
    feature_names = df.columns.drop('species')
    feature_names = feature_names._data
    target_name = 'species'
    x_train, x_test, y_train, y_test = train_test_split(df[feature_names], df[target_name], test_size=0.33,
                                                        random_state=101)
    sk_model=sk_model.fit(x_train,y_train)
    return sk_model


if __name__=='__main__':
    unittest.main(warnings='ignore')