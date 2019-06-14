import unittest

class TestImport(unittest.TestCase):

    def test_01(self):
        import nyoka
        self.assertEqual(hasattr(nyoka,"__version__"),True)
        self.assertEqual(hasattr(nyoka,"__license__"),True)
        self.assertEqual(hasattr(nyoka,"skl_to_pmml"),True)
        self.assertEqual(hasattr(nyoka,"ArimaToPMML"),True)
        self.assertEqual(hasattr(nyoka,"ExponentialSmoothingToPMML"),True)
        self.assertEqual(hasattr(nyoka,"KerasToPmml"),True)
        self.assertEqual(hasattr(nyoka,"xgboost_to_pmml"),True)
        self.assertEqual(hasattr(nyoka,"lgb_to_pmml"),True)

        import subprocess
        label = subprocess.check_output(["git", "branch"]).strip()
        for lb in label.split("\n"):
            if lb.startswith("*"):
                branch=lb.replace("* ","")
                break

        print(branch)

if __name__=='__main__':
    unittest.main(warnings='ignore')