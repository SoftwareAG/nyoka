import os
import unittest
import pandas as pd
import sys
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn_pandas import DataFrameMapper
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nyoka import skl_to_pmml
from nyoka import PMML44 as pml


class TestMethods(unittest.TestCase):
    
    def test_sklearn_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_pmml.pmml"
        model = SVC()
        pipeline_obj = Pipeline([
            ('svm',model)
        ])

        pipeline_obj.fit(irisd[features],irisd[target])
        skl_to_pmml(pipeline_obj,features,target,f_name)
        pmml_obj = pml.parse(f_name,True)
        ## 1
        svms = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine
        for mod_val, recon_val in zip(model.intercept_, svms):
            self.assertEqual("{:.16f}".format(mod_val), "{:.16f}".format(recon_val.Coefficients.absoluteValue))

        ## 2
        svm = pmml_obj.SupportVectorMachineModel[0]
        self.assertEqual(svm.RadialBasisKernelType.gamma,model._gamma)


    def test_sklearn_02(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('scaling',StandardScaler()), 
            ('knn',KNeighborsClassifier(n_neighbors = 5))
        ])

        pipeline_obj.fit(irisd[features],irisd[target])

        skl_to_pmml(pipeline_obj,features,target,"knn_pmml.pmml")

        self.assertEqual(os.path.isfile("knn_pmml.pmml"),True)

    
    def test_sklearn_03(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "rf_pmml.pmml"
        model = RandomForestClassifier(n_estimators = 100)

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
            (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()) , 
            (['petal length (cm)', 'petal width (cm)'], Imputer())
            ])),
            ("rfc", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name,True)

        ## 1
        self.assertEqual(model.n_estimators,pmml_obj.MiningModel[0].Segmentation.Segment.__len__())


    def test_sklearn_04(self):
        titanic = pd.read_csv("nyoka/tests/titanic_train.csv")

        titanic['Embarked'] = titanic['Embarked'].fillna('S')

        features = list(titanic.columns.drop(['PassengerId','Name','Ticket','Cabin','Survived']))
        target = 'Survived'

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
                (['Sex'], LabelEncoder()),
                (['Embarked'], LabelEncoder())
            ])),
            ("imp", Imputer(strategy="median")),
            ("gbc", GradientBoostingClassifier(n_estimators = 10))
        ])

        pipeline_obj.fit(titanic[features],titanic[target])

        skl_to_pmml(pipeline_obj, features, target, "gb_pmml.pmml")

        self.assertEqual(os.path.isfile("gb_pmml.pmml"),True)


    def test_sklearn_05(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg'],axis=1)
        y = df['mpg']

        features = [name for name in df.columns if name not in ('mpg')]
        target = 'mpg'

        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', TfidfVectorizer())
            ])),
            ('model',DecisionTreeRegressor())
        ])

        pipeline_obj.fit(X,y)
        
        skl_to_pmml(pipeline_obj,features,target,"dtr_pmml.pmml")

        self.assertEqual(os.path.isfile("dtr_pmml.pmml"),True)


    def test_sklearn_06(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'
        f_name = "linearregression_pmml.pmml"
        model = LinearRegression()

        pipeline_obj = Pipeline([
            ('model',model)
        ])

        pipeline_obj.fit(X,y)
        skl_to_pmml(pipeline_obj,features,target,f_name)
        pmml_obj = pml.parse(f_name, True)

        ## 1
        reg_tab = pmml_obj.RegressionModel[0].RegressionTable[0]
        self.assertEqual(reg_tab.intercept,model.intercept_)

        ## 2
        for model_val, pmml_val in zip(model.coef_, reg_tab.NumericPredictor):
            self.assertEqual("{:.16f}".format(model_val),"{:.16f}".format(pmml_val.coefficient))


    def test_sklearn_07(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "logisticregression_pmml.pmml"
        model = LogisticRegression()

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
            (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()) , 
            (['petal length (cm)', 'petal width (cm)'], Imputer())
            ])),
            ("lr", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name,True)

        ## 1
        segmentation = pmml_obj.MiningModel[0].Segmentation
        self.assertEqual(segmentation.Segment.__len__(), model.classes_.__len__()+1)

        ## 2


    def test_sklearn_08(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('pca',PCA(2)),
            ('mod',LogisticRegression())
        ])
        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "logisticregression_pca_pmml.pmml")

        self.assertEqual(os.path.isfile("logisticregression_pca_pmml.pmml"),True)


    def test_sklearn_09(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("SGD", SGDClassifier())
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "sgdclassifier_pmml.pmml")

        self.assertEqual(os.path.isfile("sgdclassifier_pmml.pmml"),True)


    def test_sklearn_10(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("lsvc", LinearSVC())
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "linearsvc_pmml.pmml")

        self.assertEqual(os.path.isfile("linearsvc_pmml.pmml"),True)


    def test_sklearn_11(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        pipeline_obj = Pipeline([
            ('model',LinearSVR())
        ])

        pipeline_obj.fit(X,y)

        skl_to_pmml(pipeline_obj,features,target,"linearsvr_pmml.pmml")

        self.assertEqual(os.path.isfile("linearsvr_pmml.pmml"),True)


    def test_sklearn_12(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        pipeline_obj = Pipeline([
            ('model',GradientBoostingRegressor())
        ])

        pipeline_obj.fit(X,y)

        skl_to_pmml(pipeline_obj,features,target,"gbr.pmml")

        self.assertEqual(os.path.isfile("gbr.pmml"),True)


    def test_sklearn_13(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("SGD", DecisionTreeClassifier())
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "dtr_clf.pmml")

        self.assertEqual(os.path.isfile("dtr_clf.pmml"),True)


    def test_sklearn_14(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        pipeline_obj = Pipeline([
            ('model',RandomForestRegressor())
        ])

        pipeline_obj.fit(X,y)

        skl_to_pmml(pipeline_obj,features,target,"rfr.pmml")

        self.assertEqual(os.path.isfile("rfr.pmml"),True)


    def test_sklearn_15(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        pipeline_obj = Pipeline([
            ('model',KNeighborsRegressor())
        ])

        pipeline_obj.fit(X,y)

        skl_to_pmml(pipeline_obj,features,target,"knnr.pmml")

        self.assertEqual(os.path.isfile("knnr.pmml"),True)


    def test_sklearn_16(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        pipeline_obj = Pipeline([
            ('model',SVR())
        ])

        pipeline_obj.fit(X,y)

        skl_to_pmml(pipeline_obj,features,target,"svr.pmml")

        self.assertEqual(os.path.isfile("svr.pmml"),True)


    def test_sklearn_17(self):
        irisdata = datasets.load_iris()
        iris = pd.DataFrame(irisdata.data,columns=irisdata.feature_names)
        iris['Species'] = irisdata.target

        feature_names = iris.columns.drop('Species')

        X = iris[iris.columns.drop(['Species'])]
        
        pipeline_obj = Pipeline([
            ('standard_scaler',StandardScaler()),
            ('Imputer',Imputer()),
            ('model',OneClassSVM())
        ])

        pipeline_obj.fit(X)
        skl_to_pmml(pipeline_obj, feature_names, pmml_f_name="one_class_svm.pmml")
        self.assertEqual(os.path.isfile("one_class_svm.pmml"),True)


    def test_sklearn_18(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("model", GaussianNB())
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "gnb.pmml")

        self.assertEqual(os.path.isfile("gnb.pmml"),True)


    def test_sklearn_19(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("model", SGDClassifier())
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "sgdc.pmml")

        self.assertEqual(os.path.isfile("sgdc.pmml"),True)


    def test_sklearn_20(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("model", RidgeClassifier())
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "ridge.pmml")

        self.assertEqual(os.path.isfile("ridge.pmml"),True)


    def test_sklearn_21(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("model", LinearDiscriminantAnalysis())
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "lda.pmml")

        self.assertEqual(os.path.isfile("lda.pmml"),True)


if __name__=='__main__':
    print(os.listdir())
    unittest.main(warnings='ignore')







