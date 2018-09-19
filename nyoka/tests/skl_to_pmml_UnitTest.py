import os
import unittest
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nyoka import skl_to_pmml


class TestMethods(unittest.TestCase):

    
    def test_sklearn_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data,columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'

        pipeline_obj = Pipeline([
            ('svm',SVC())
        ])

        pipeline_obj.fit(irisd[features],irisd[target])

        skl_to_pmml(pipeline_obj,features,target,"svc_pmml.pmml")

        self.assertEqual(os.path.isfile("svc_pmml.pmml"),True)


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

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
            (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()) , 
            (['petal length (cm)', 'petal width (cm)'], Imputer())
            ])),
            ("rfc", RandomForestClassifier(n_estimators = 100))
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "rf_pmml.pmml")

        self.assertEqual(os.path.isfile("rf_pmml.pmml"),True)


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

if __name__=='__main__':
    unittest.main(warnings='ignore')







