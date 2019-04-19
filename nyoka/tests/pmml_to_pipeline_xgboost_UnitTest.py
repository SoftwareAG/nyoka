from nyoka.reconstruct.pmml_to_pipeline import generate_skl_pipeline
import unittest

class TestMethods(unittest.TestCase):

    def test_xgboost_01(self):
        recon_pipeline = generate_skl_pipeline("xgbc_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"XGBClassifier")


    def test_xgboost_02(self):
        recon_pipeline = generate_skl_pipeline("xgbr_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"XGBRegressor")


    def test_xgboost_03(self):
        recon_pipeline = generate_skl_pipeline("xgbc_pmml_preprocess.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"XGBClassifier")


    def test_xgboost_04(self):
        recon_pipeline = generate_skl_pipeline("xgbr_pmml_preprocess.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"XGBRegressor")


if __name__=='__main__':
    unittest.main(warnings='ignore')