import os
import unittest
import pandas as pd
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
from nyoka import model_to_pmml


class TestMethods(unittest.TestCase):
    
    def test_sklearn_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        svm = SVC()
        svm.fit(irisd[features],irisd[target])

        pmml_file_name = "svc_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':svm,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_02(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('scaling',StandardScaler())
        ])

        knn = KNeighborsClassifier(n_neighbors = 5)
        X = pipeline_obj.fit_transform(irisd[features])
        knn.fit(X,irisd[target])

        pmml_file_name = "knn_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':knn,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)

    
    def test_sklearn_03(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
            (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()) , 
            (['petal length (cm)', 'petal width (cm)'], Imputer())
            ]))
        ])

        X = pipeline_obj.fit_transform(irisd[features])

        rfc = RandomForestClassifier(n_estimators = 100)
        rfc.fit(X,irisd[target])

        pmml_file_name = "rf_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':rfc,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


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
            ("imp", Imputer(strategy="median"))
        ])

        X = pipeline_obj.fit_transform(titanic[features])

        gbc = GradientBoostingClassifier(n_estimators = 10)
        gbc.fit(X,titanic[target])

        pmml_file_name = "gb_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':gbc,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_05(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg'],axis=1)
        y = df['mpg']

        features = [name for name in df.columns if name not in ('mpg')]
        target = 'mpg'

        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', TfidfVectorizer())
            ]))
        ])

        X = pipeline_obj.fit_transform(X)

        model = DecisionTreeRegressor()
        model.fit(X,y)

        pmml_file_name = "dtr_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_06(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        model = LinearRegression()
        model.fit(X,y)

        pmml_file_name = "linearregression_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_07(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
            (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()) , 
            (['petal length (cm)', 'petal width (cm)'], Imputer())
            ]))
        ])

        X = pipeline_obj.fit_transform(irisd[features])
        
        lr = LogisticRegression()
        lr.fit(X,irisd[target])

        pmml_file_name = "logisticregression_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':lr,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_08(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('pca',PCA(2))
        ])

        X = pipeline_obj.fit_transform(irisd[features])

        mod = LogisticRegression()
        mod.fit(X,irisd[target])

        pmml_file_name = "logisticregression_pca_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':mod,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_09(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        SGD = SGDClassifier()
        SGD.fit(irisd[features],irisd[target])

        pmml_file_name = "sgdclassifier_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':SGD,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_10(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        lsvc = LinearSVC()
        lsvc.fit(irisd[features], irisd[target])

        pmml_file_name = "linearsvc_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':lsvc,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_11(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        model = LinearSVR()
        model.fit(X,y)

        pmml_file_name = "linearsvr_pmml.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_12(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        model = GradientBoostingRegressor()
        model.fit(X,y)

        pmml_file_name = "gbr.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_13(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        SGD = DecisionTreeClassifier()

        SGD.fit(irisd[features], irisd[target])

        pmml_file_name = "dtr_clf.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':SGD,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_14(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        model = RandomForestRegressor()
        model.fit(X,y)

        pmml_file_name = "rfr.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_15(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        model = KNeighborsRegressor()
        model.fit(X,y)

        pmml_file_name = "knnr.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_16(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg','car name'],axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'

        model = SVR()
        model.fit(X,y)

        pmml_file_name = "svr.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_17(self):
        irisdata = datasets.load_iris()
        iris = pd.DataFrame(irisdata.data,columns=irisdata.feature_names)
        iris['Species'] = irisdata.target

        feature_names = iris.columns.drop('Species')

        X = iris[iris.columns.drop(['Species'])]
        
        pipeline_obj = Pipeline([
            ('standard_scaler',StandardScaler()),
            ('Imputer',Imputer())
        ])
        X = pipeline_obj.fit_transform(X)
        model = OneClassSVM()
        model.fit(X)

        pmml_file_name = "one_class_svm.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':pipeline_obj,
                'modelObj':model,
                'featuresUsed':feature_names,
                'targetName':None,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_18(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        model = GaussianNB()
        model.fit(irisd[features], irisd[target])

        pmml_file_name = "gnb.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)

    def test_sklearn_19(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        model = SGDClassifier()
        model.fit(irisd[features], irisd[target])

        pmml_file_name = "sgdc.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_20(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        model = RidgeClassifier()
        model.fit(irisd[features], irisd[target])

        pmml_file_name = "ridge.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


    def test_sklearn_21(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        model = LinearDiscriminantAnalysis()
        model.fit(irisd[features], irisd[target])

        pmml_file_name = "lda.pmml"
        toExportDict={
            'model1':{
                'hyperparameters':None,
                'preProcessingScript':None,
                'pipelineObj':None,
                'modelObj':model,
                'featuresUsed':features,
                'targetName':target,
                'postProcessingScript':None,
                'taskType': 'score'
            }
        }
        model_to_pmml(toExportDict, pmml_f_name=pmml_file_name)

        self.assertEqual(os.path.isfile(pmml_file_name),True)


if __name__=='__main__':
    unittest.main(warnings='ignore')







