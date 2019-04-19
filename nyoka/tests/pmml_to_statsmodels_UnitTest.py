from nyoka.reconstruct.pmml_to_statsmodels import generate_statsmodels
import unittest

class TestMethods(unittest.TestCase):
    
    def test_arima_01(self):
        result_obj, model_obj = generate_statsmodels('non_seasonal_arima1.pmml')
        self.assertEqual(model_obj.__class__.__name__,"ARIMA")
        self.assertEqual(result_obj.__class__.__name__,"ARIMAResultsWrapper")


    def test_non_seasonal_arima_01(self):
        result_obj, model_obj = generate_statsmodels('seasonal_arima1.pmml')
        self.assertEqual(model_obj.__class__.__name__,"SARIMAX")
        self.assertEqual(result_obj.__class__.__name__,"SARIMAXResultsWrapper")


if __name__=='__main__':
    unittest.main(warnings='ignore')