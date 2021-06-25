import unittest,sys

class TestImport(unittest.TestCase):

    def test_01(self):
        import nyoka
        self.assertEqual(hasattr(nyoka,"__version__"),True)
        self.assertEqual(hasattr(nyoka,"__license__"),True)
        self.assertEqual(hasattr(nyoka,"skl_to_pmml"),True)
        self.assertEqual(hasattr(nyoka,"ExponentialSmoothingToPMML"),True)
        self.assertEqual(hasattr(nyoka,"xgboost_to_pmml"),True)
        self.assertEqual(hasattr(nyoka,"lgb_to_pmml"),True)
        self.assertEqual(hasattr(nyoka,"StatsmodelsToPmml"),True)

if __name__=='__main__':
    unittest.main(warnings='ignore')