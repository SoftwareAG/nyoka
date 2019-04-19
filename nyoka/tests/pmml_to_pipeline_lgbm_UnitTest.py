from nyoka.reconstruct.pmml_to_pipeline import generate_skl_pipeline
import unittest

class TestMethods(unittest.TestCase):

    def test_lgbm_01(self):
        recon_pipeline = generate_skl_pipeline("lgbmc_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"LGBMClassifier")


    def test_lgbm_02(self):
        recon_pipeline = generate_skl_pipeline("lgbmr_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"LGBMRegressor")


    def test_lgbm_03(self):
        recon_pipeline = generate_skl_pipeline("lgbmc_pmml_preprocess.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"LGBMClassifier")


    def test_lgbm_04(self):
        recon_pipeline = generate_skl_pipeline("lgbmr_pmml_preprocess.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"LGBMRegressor")


if __name__=='__main__':
    unittest.main(warnings='ignore')