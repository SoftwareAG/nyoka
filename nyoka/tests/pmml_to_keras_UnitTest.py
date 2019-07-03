from nyoka.reconstruct.pmml_to_keras_model import GenerateKerasModel as generate_keras_model
from nyoka import PMML43Ext as ny
import unittest

class TestMethods(unittest.TestCase):

    def test_keras_01(self):
        recon_model = generate_keras_model("2classMBNet.pmml")
        self.assertEqual(len(recon_model.model.layers), len(recon_model.nyoka_pmml.DeepNetwork[0].NetworkLayer))


    def test_keras_02(self):
        recon_model = generate_keras_model("sequentialModel.pmml")
        self.assertEqual(len(recon_model.model.layers), len(recon_model.nyoka_pmml.DeepNetwork[0].NetworkLayer))


if __name__=='__main__':
    unittest.main(warnings='ignore')