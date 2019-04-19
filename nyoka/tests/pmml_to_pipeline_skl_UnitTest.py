from nyoka.reconstruct.pmml_to_pipeline import generate_skl_pipeline
import unittest

class TestMethods(unittest.TestCase):

    def test_sklearn_01(self):
        recon_pipeline = generate_skl_pipeline("svc_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"SVC")


    def test_sklearn_02(self):
        recon_pipeline = generate_skl_pipeline("knn_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"KNeighborsClassifier")


    def test_sklearn_03(self):
        recon_pipeline = generate_skl_pipeline("rf_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"RandomForestClassifier")


    def test_sklearn_04(self):
        recon_pipeline = generate_skl_pipeline("gb_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"GradientBoostingClassifier")


    def test_sklearn_05(self):
        recon_pipeline = generate_skl_pipeline("dtr_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"DecisionTreeRegressor")


    def test_sklearn_06(self):
        recon_pipeline = generate_skl_pipeline("linearregression_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"LinearRegression")


    def test_sklearn_07(self):
        recon_pipeline = generate_skl_pipeline("logisticregression_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"LogisticRegression")


    def test_sklearn_08(self):
        recon_pipeline = generate_skl_pipeline("sgdclassifier_pmml.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"SGDClassifier")


    # def test_sklearn_09(self):
    #     recon_pipeline = generate_skl_pipeline("linearsvc_pmml.pmml")
    #     model_obj = recon_pipeline.steps[-1][-1]
    #     self.assertEqual(model_obj.__class__.__name__,"LinearSVC")


    # def test_sklearn_10(self):
    #     recon_pipeline = generate_skl_pipeline("linearsvr_pmml.pmml")
    #     model_obj = recon_pipeline.steps[-1][-1]
    #     self.assertEqual(model_obj.__class__.__name__,"LinearSVR")


    def test_sklearn_11(self):
        recon_pipeline = generate_skl_pipeline("dtr_clf.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"DecisionTreeClassifier")


    def test_sklearn_12(self):
        recon_pipeline = generate_skl_pipeline("gbr.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"GradientBoostingRegressor")


    def test_sklearn_13(self):
        recon_pipeline = generate_skl_pipeline("rfr.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"RandomForestRegressor")


    def test_sklearn_14(self):
        recon_pipeline = generate_skl_pipeline("knnr.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"KNeighborsRegressor")


    def test_sklearn_15(self):
        recon_pipeline = generate_skl_pipeline("svr.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"SVR")


    def test_sklearn_16(self):
        recon_pipeline = generate_skl_pipeline("gnb.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"GaussianNB")


    def test_sklearn_17(self):
        recon_pipeline = generate_skl_pipeline("sgdc.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"SGDClassifier")


    def test_sklearn_18(self):
        recon_pipeline = generate_skl_pipeline("ridge.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"RidgeClassifier")


    def test_sklearn_19(self):
        recon_pipeline = generate_skl_pipeline("lda.pmml")
        model_obj = recon_pipeline.steps[-1][-1]
        self.assertEqual(model_obj.__class__.__name__,"LinearDiscriminantAnalysis")


    # def test_sklearn_20(self):
    #     recon_pipeline = generate_skl_pipeline("logisticregression_pca_pmml.pmml")
    #     model_obj = recon_pipeline.steps[-1][-1]
    #     self.assertEqual(model_obj.__class__.__name__,"LogisticRegression")


if __name__=='__main__':
    unittest.main(warnings='ignore')