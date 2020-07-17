import sys, os

import unittest
import pandas as pd
import numpy
import sys
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, MinMaxScaler, MaxAbsScaler, \
    RobustScaler, \
    Binarizer, PolynomialFeatures, OneHotEncoder, KBinsDiscretizer
try:
    from sklearn.preprocessing import Imputer
except:
    from sklearn.impute import SimpleImputer as Imputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn_pandas import DataFrameMapper
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from nyoka.preprocessing import Lag

from nyoka import skl_to_pmml
from nyoka import PMML44 as pml
from base.enums import *
from collections import Counter


class TestMethods(unittest.TestCase):

    def parse_nodes(self, node, values, scores):
        if node.SimplePredicate.operator == "lessOrEqual":
            values.append(node.SimplePredicate.value)
        else:
            values.append(-2)
        if len(node.Node) > 0:
            scores.append(-2)
        else:
            scores.append(node.score)
        for nd in node.Node:
            self.parse_nodes(nd, values, scores)

    def test_sklearn_01(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_pmml.pmml"
        model = SVC()
        pipeline_obj = Pipeline([
            ('svm', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, model_name="testModel")
        pmml_obj = pml.parse(f_name, True)

        # 1
        svms = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine
        for mod_val, recon_val in zip(model.intercept_, svms):
            self.assertEqual("{:.12f}".format(mod_val), "{:.12f}".format(recon_val.Coefficients.absoluteValue))

        # 2
        svm = pmml_obj.SupportVectorMachineModel[0]
        self.assertEqual("{:.12f}".format(model._gamma), "{:.12f}".format(float(svm.RadialBasisKernelType.gamma)))

    def test_sklearn_02(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_euclidean.pmml"
        model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.euclidean)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual(model.effective_metric_,
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.euclidean.__class__.__name__)

        # 5
        self.assertEqual(model.weights, pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_03(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_minkowski.pmml"
        model = KNeighborsClassifier(p=3, n_neighbors=5, weights="distance")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.minkowski)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual(model.effective_metric_,
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.minkowski.__class__.__name__)

        # 5
        self.assertEqual(model.weights, pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_04(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_cityblock.pmml"
        model = KNeighborsClassifier(n_neighbors=5, metric="cityblock", weights="distance")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.cityBlock)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual(model.effective_metric_,
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.cityBlock.__class__.__name__.lower())

        # 5
        self.assertEqual(model.weights, pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_05(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_manhattan.pmml"
        model = KNeighborsClassifier(n_neighbors=5, metric="manhattan", weights="distance")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.cityBlock)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual("cityBlock",
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.cityBlock.__class__.__name__)

        # 5
        self.assertEqual(model.weights, pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_06(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_chebyshev.pmml"
        model = KNeighborsClassifier(n_neighbors=5, metric="chebyshev", weights="distance")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.chebychev)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual("chebychev",
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.chebychev.__class__.__name__)

        # 5
        self.assertEqual(model.weights, pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_07(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_matching.pmml"
        model = KNeighborsClassifier(n_neighbors=5, metric="matching", weights="uniform")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.simpleMatching)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual("simpleMatching",
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.simpleMatching.__class__.__name__)

        # 5
        self.assertEqual("similarity", pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_08(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_jaccard.pmml"
        model = KNeighborsClassifier(n_neighbors=5, metric="jaccard", weights="uniform")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.jaccard)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual(model.effective_metric_,
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.jaccard.__class__.__name__)

        # 5
        self.assertEqual("similarity", pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_09(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_rogerstanimoto.pmml"
        model = KNeighborsClassifier(n_neighbors=5, metric="rogerstanimoto", weights="uniform")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.tanimoto)

        # 3
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 4
        self.assertEqual("tanimoto",
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.tanimoto.__class__.__name__)

        # 5
        self.assertEqual("similarity", pmml_obj.NearestNeighborModel[0].ComparisonMeasure.kind)

    def test_sklearn_10(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "rf_pmml.pmml"
        model = RandomForestClassifier(n_estimators=100)

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
                (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()),
                (['petal length (cm)', 'petal width (cm)'], Imputer())
            ])),
            ("rfc", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 2
        segments = pmml_obj.MiningModel[0].Segmentation.Segment
        estms = model.estimators_
        self.assertEqual(len(segments), len(estms))
        for segment, estm in zip(segments, estms):
            values = []
            scores = []
            for nd in segment.TreeModel.Node.Node:
                self.parse_nodes(nd, values, scores)
            values.append(-2)
            scores.insert(0, -2)
            for a, b in zip(scores, estm.tree_.value):
                if a == -2:
                    continue
                self.assertEqual(a, str(numpy.argmax(b[0])))
            for a, b in zip(values, estm.tree_.threshold):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))

        # 4
        self.assertEqual(os.path.isfile(f_name), True)

        # 5
        self.assertEqual(model.n_estimators, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 6
        self.assertEqual(MULTIPLE_MODEL_METHOD.AVERAGE.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

    def test_sklearn_11(self):
        titanic = pd.read_csv("nyoka/tests/titanic_train.csv")
        features = titanic.columns
        target = 'Survived'
        f_name = "gb_pmml.pmml"
        model = GradientBoostingClassifier(n_estimators=10)
        pipeline_obj = Pipeline([
            ("imp", Imputer(strategy="median")),
            ("gbc", model)
        ])

        pipeline_obj.fit(titanic[features], titanic[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 2
        segments = pmml_obj.MiningModel[0].Segmentation.Segment[0].MiningModel.Segmentation.Segment
        estms = model.estimators_.ravel()
        self.assertEqual(len(segments), len(estms))
        for segment, estm in zip(segments, estms):
            values = []
            scores = []
            for nd in segment.TreeModel.Node.Node:
                self.parse_nodes(nd, values, scores)
            values.append(-2)
            scores.insert(0, -2)
            for a, b in zip(scores, estm.tree_.value.ravel()):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
            for a, b in zip(values, estm.tree_.threshold):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))


        # 3
        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

        # 4
        self.assertEqual(model.min_samples_split, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 5
        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[1].RegressionModel.normalizationMethod)

    def test_sklearn_12(self):
        titanic = pd.read_csv("nyoka/tests/titanic_train.csv")
        features = titanic.columns
        target = 'Survived'
        f_name = "gb_pmml.pmml"
        model = GradientBoostingClassifier(n_estimators=10, criterion="mse")
        pipeline_obj = Pipeline([
            ("imp", Imputer(strategy="most_frequent")),
            ("gbc", model)
        ])

        pipeline_obj.fit(titanic[features], titanic[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 2
        segments = pmml_obj.MiningModel[0].Segmentation.Segment[0].MiningModel.Segmentation.Segment
        estms = model.estimators_.ravel()
        self.assertEqual(len(segments), len(estms))
        for segment, estm in zip(segments, estms):
            values = []
            scores = []
            for nd in segment.TreeModel.Node.Node:
                self.parse_nodes(nd, values, scores)
            values.append(-2)
            scores.insert(0, -2)
            for a, b in zip(scores, estm.tree_.value.ravel()):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
            for a, b in zip(values, estm.tree_.threshold):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))


        # 3
        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

        # 4
        self.assertEqual(model.min_samples_split, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 5
        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[1].RegressionModel.normalizationMethod)


    def test_sklearn_14(self):
        titanic = pd.read_csv("nyoka/tests/titanic_train.csv")
        features = titanic.columns
        target = 'Survived'
        f_name = "gb_pmml.pmml"
        model = GradientBoostingClassifier(n_estimators=10, criterion="mae")
        pipeline_obj = Pipeline([
            ("imp", Imputer(strategy="median")),
            ("gbc", model)
        ])

        pipeline_obj.fit(titanic[features], titanic[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 2
        segments = pmml_obj.MiningModel[0].Segmentation.Segment[0].MiningModel.Segmentation.Segment
        estms = model.estimators_.ravel()
        self.assertEqual(len(segments), len(estms))
        for segment, estm in zip(segments, estms):
            values = []
            scores = []
            for nd in segment.TreeModel.Node.Node:
                self.parse_nodes(nd, values, scores)
            values.append(-2)
            scores.insert(0, -2)
            for a, b in zip(scores, estm.tree_.value.ravel()):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
            for a, b in zip(values, estm.tree_.threshold):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))


        # 3
        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

        # 4
        self.assertEqual(model.min_samples_split, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 5
        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[1].RegressionModel.normalizationMethod)

    def test_sklearn_15(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg'], axis=1)
        y = df['mpg']

        features = [name for name in df.columns if name not in ('mpg')]
        target = 'mpg'
        f_name = "dtr_pmml.pmml"
        model = DecisionTreeRegressor()
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', TfidfVectorizer())
            ])),
            ('model', model)
        ])
        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse("dtr_pmml.pmml", True)

        values = []
        scores = []
        for nd in pmml_obj.TreeModel[0].Node.Node:
            self.parse_nodes(nd, values, scores)
        values.append(-2)
        scores.insert(0, -2)
        for a, b in zip(scores, model.tree_.value.ravel()):
            if a == -2:
                continue
            self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
        for a, b in zip(values, model.tree_.threshold):
            if a == -2:
                continue
            self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))

        # 4
        self.assertEqual(os.path.isfile(f_name), True)

    def test_sklearn_16(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg', 'car name'], axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'
        f_name = "linearregression_pmml.pmml"
        model = LinearRegression()

        pipeline_obj = Pipeline([
            ('model', model)
        ])

        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        reg_tab = pmml_obj.RegressionModel[0].RegressionTable[0]
        self.assertEqual(reg_tab.intercept, model.intercept_)

        # 2
        for model_val, pmml_val in zip(model.coef_, reg_tab.NumericPredictor):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

    def test_sklearn_17(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "logisticregression_pmml.pmml"
        model = LogisticRegression()

        pipeline_obj = Pipeline([
            ("mapping", DataFrameMapper([
                (['sepal length (cm)', 'sepal width (cm)'], StandardScaler()),
                (['petal length (cm)', 'petal width (cm)'], Imputer())
            ])),
            ("lr", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3
        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4
        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        seg_tab = pmml_obj.MiningModel[0].Segmentation.Segment
        for model_coef, pmml_seg in zip(model.coef_, seg_tab):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7
        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_18(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = [i % 2 for i in range(iris.data.shape[0])]

        features = irisd.columns.drop('Species')
        target = 'Species'
        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ('pca', PCA(2)),
            ('mod', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        skl_to_pmml(pipeline_obj, features, target, "logisticregression_pca_pmml.pmml")

        pmml_obj = pml.parse("logisticregression_pca_pmml.pmml", True)

        reg_tab = pmml_obj.RegressionModel[0].RegressionTable[0]

        # 1
        self.assertEqual(os.path.isfile("logisticregression_pca_pmml.pmml"), True)

        # 2

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.RegressionModel[0].normalizationMethod)

        # 3
        self.assertEqual('{:.9g}'.format(model.intercept_[0]), '{:.9g}'.format(reg_tab.intercept))

        # 4
        model_coef_reshape = model.coef_.reshape(2, 1)
        for model_val, pmml_val in zip(model_coef_reshape, reg_tab.NumericPredictor):
            self.assertEqual("{:.12f}".format(model_val[0]), "{:.12f}".format(pmml_val.coefficient))

    def test_sklearn_19(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "sgdclassifier_pmml.pmml"

        model = SGDClassifier()
        pipeline_obj = Pipeline([
            ("SGD", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation
        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

        # 3
        self.assertEqual(4, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        seg_tab = pmml_obj.MiningModel[0].Segmentation.Segment
        for model_coef, pmml_seg in zip(model.coef_, seg_tab):
            num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
            for model_val, pmml_val in zip(model_coef, num_predict):
                self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_20(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "linearsvc_pmml.pmml"

        model = LinearSVC()
        pipeline_obj = Pipeline([
            ("lsvc", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

        # 3
        self.assertEqual(4, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 4
        sm = pmml_obj.MiningModel[0].Segmentation.Segment
        for mod_val, recon_val in zip(model.intercept_, sm):
            self.assertEqual("{:.12f}".format(mod_val),
                             "{:.12f}".format(recon_val.RegressionModel.RegressionTable[0].intercept))

        # 5
        lin_tab = pmml_obj.MiningModel[0].Segmentation
        for mod_val, pmml_val in zip(model.coef_, lin_tab.Segment):
            num_pred = pmml_val.RegressionModel.RegressionTable[0].NumericPredictor
            for model_val, pmml_val in zip(mod_val, num_pred):
                self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

    def test_sklearn_21(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg', 'car name'], axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'
        f_name = "linearsvr_pmml.pmml"

        model = LinearSVR()
        pipeline_obj = Pipeline([
            ('model', model)
        ])

        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual("{:.12f}".format(model.intercept_[0]),
                         "{:.12f}".format(pmml_obj.RegressionModel[0].RegressionTable[0].intercept))

        # 3
        reg_tab = pmml_obj.RegressionModel[0].RegressionTable[0].NumericPredictor
        for model_val, pmml_val in zip(model.coef_, reg_tab):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

    def test_sklearn_22(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg', 'car name'], axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'
        f_name = "gbr.pmml"

        model = GradientBoostingRegressor()
        pipeline_obj = Pipeline([
            ('model', model)
        ])

        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segments = pmml_obj.MiningModel[0].Segmentation.Segment
        estms = model.estimators_.ravel()
        self.assertEqual(len(segments), len(estms))
        for segment, estm in zip(segments, estms):
            values = []
            scores = []
            for nd in segment.TreeModel.Node.Node:
                self.parse_nodes(nd, values, scores)
            values.append(-2)
            scores.insert(0, -2)
            for a, b in zip(scores, estm.tree_.value.ravel()):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
            for a, b in zip(values, estm.tree_.threshold):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))

        # 3
        self.assertEqual(os.path.isfile(f_name), True)

        # 4
        self.assertEqual(model.n_estimators_,
                         pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 5

        self.assertEqual(MULTIPLE_MODEL_METHOD.SUM.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

    def test_sklearn_23(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "dtr_clf.pmml"

        model = DecisionTreeClassifier()
        pipeline_obj = Pipeline([
            ("SGD", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        values = []
        scores = []
        for nd in pmml_obj.TreeModel[0].Node.Node:
            self.parse_nodes(nd, values, scores)
        values.append(-2)
        scores.insert(0, -2)
        for a, b in zip(scores, model.tree_.value):
            if a == -2:
                continue
            self.assertEqual(a, str(numpy.argmax(b[0])))
        for a, b in zip(values, model.tree_.threshold):
            if a == -2:
                continue
            self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))

        # 5
        self.assertEqual(os.path.isfile(f_name), True)

    def test_sklearn_24(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg', 'car name'], axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'
        f_name = "rfr.pmml"

        model = RandomForestRegressor()
        pipeline_obj = Pipeline([
            ('model', model)
        ])

        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segments = pmml_obj.MiningModel[0].Segmentation.Segment
        estms = model.estimators_
        self.assertEqual(len(segments), len(estms))
        for segment, estm in zip(segments, estms):
            values = []
            scores = []
            for nd in segment.TreeModel.Node.Node:
                self.parse_nodes(nd, values, scores)
            values.append(-2)
            scores.insert(0, -2)
            for a, b in zip(scores, estm.tree_.value.ravel()):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
            for a, b in zip(values, estm.tree_.threshold):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))

        # 3
        self.assertEqual(os.path.isfile(f_name), True)

        # 4
        self.assertEqual(model.n_estimators, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 5

        self.assertEqual(MULTIPLE_MODEL_METHOD.AVERAGE.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

    def test_sklearn_25(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "knn_pmml_euclidean_r.pmml"
        model = KNeighborsRegressor(n_neighbors=5, weights="distance")
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('knn', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name, description="A test model")
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertIsNotNone(pmml_obj.NearestNeighborModel[0].ComparisonMeasure.euclidean)

        # 4
        self.assertEqual(model.n_neighbors, pmml_obj.NearestNeighborModel[0].numberOfNeighbors)

        # 5
        self.assertEqual(model.effective_metric_,
                         pmml_obj.NearestNeighborModel[0].ComparisonMeasure.euclidean.__class__.__name__)

    def test_sklearn_26(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg', 'car name'], axis=1)
        y = df['mpg']

        features = X.columns
        target = 'mpg'
        model = SVR()
        f_name = "svr.pmml"

        pipeline_obj = Pipeline([
            ('model', model)
        ])

        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        svm = pmml_obj.SupportVectorMachineModel[0]
        self.assertEqual("{:.12f}".format(model._gamma), "{:.12f}".format(svm.RadialBasisKernelType.gamma))

        # 3
        svm_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine[0]
        self.assertEqual(model.intercept_[0], svm_tab.Coefficients.absoluteValue)

        # 4
        vect_tab = pmml_obj.SupportVectorMachineModel[0].VectorDictionary.VectorInstance
        for model_vectors, pmml_vertors in zip(model.support_vectors_, vect_tab):
            entries_val = pmml_vertors.REAL_SparseArray.REAL_Entries
            arr = numpy.array(entries_val)
            arr = numpy.array(entries_val, dtype=numpy.float64)
            for model_val, pmml_val in zip(model_vectors, arr):
                self.assertEqual(model_val, pmml_val)

        # 5
        for model_val, pmml_val in zip(model.dual_coef_[0], svm_tab.Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_27(self):
        irisdata = datasets.load_iris()
        iris = pd.DataFrame(irisdata.data, columns=irisdata.feature_names)
        iris['Species'] = irisdata.target

        feature_names = iris.columns.drop('Species')

        X = iris[iris.columns.drop(['Species'])]
        model = OneClassSVM(gamma=0.25)
        pipeline_obj = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('Imputer', Imputer()),
            ('model', model)
        ])

        pipeline_obj.fit(X)
        skl_to_pmml(pipeline_obj, feature_names, pmml_f_name="one_class_svm.pmml")
        pmml_obj = pml.parse("one_class_svm.pmml", True)

        # 1
        self.assertEqual(os.path.isfile("one_class_svm.pmml"), True)

        # 2
        svm_tab = pmml_obj.AnomalyDetectionModel[0].SupportVectorMachineModel
        self.assertEqual(model.gamma, svm_tab.RadialBasisKernelType.gamma)

        # 3
        self.assertEqual(model.intercept_[0], svm_tab.SupportVectorMachine[0].Coefficients.absoluteValue)

        # 4
        for model_val, pmml_val in zip(model.dual_coef_[0], svm_tab.SupportVectorMachine[0].Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_28(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "gnb.pmml"
        model = GaussianNB()
        pipeline_obj = Pipeline([
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        target_tab = pmml_obj.NaiveBayesModel[0].BayesOutput.TargetValueCounts.TargetValueCount
        for model_val, pmml_val in zip(model.class_count_, target_tab):
            self.assertEqual(model_val, pmml_val.count)

        # 3
        the_tab = model.theta_.transpose()
        sig_tab = model.sigma_.transpose()
        bay_tab = pmml_obj.NaiveBayesModel[0].BayesInputs.BayesInput
        for model_the_val, model_sig_val, pmml_bay_val in zip(the_tab, sig_tab, bay_tab):
            for the_val, sig_val, tar_val in zip(model_the_val, model_sig_val,
                                                 pmml_bay_val.TargetValueStats.TargetValueStat):
                self.assertEqual("{:.12f}".format(the_val), "{:.12f}".format(tar_val.GaussianDistribution.mean))
                self.assertEqual("{:.12f}".format(sig_val), "{:.12f}".format(tar_val.GaussianDistribution.variance))

    def test_sklearn_29(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "ridge.pmml"

        model = RidgeClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        #  3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        #  4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        #  5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_30(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "lda.pmml"

        model = LinearDiscriminantAnalysis()
        pipeline_obj = Pipeline([
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_31(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "binarizer.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("scaler", Binarizer(threshold=2)),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_32(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "minmax.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_33(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "robust.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("scaler", RobustScaler()),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_34(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "maxabs.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_35(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        irisd['new'] = [i % 3 for i in range(iris.data.shape[0])]
        irisd.to_csv("test_new.csv", index=False)

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "labelencoder.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("mapper", DataFrameMapper([
                (["new"], LabelEncoder()),
                (iris.feature_names, None)
            ])),
            ('scale', StandardScaler()),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_36(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        irisd['new'] = [i % 3 for i in range(iris.data.shape[0])]
        irisd.to_csv("test_new.csv", index=False)

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "labelbinarizer.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("mapper", DataFrameMapper([
                (["new"], LabelBinarizer()),
                (iris.feature_names, None)
            ])),
            ('scale', StandardScaler()),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_37(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        irisd['new'] = [i % 3 for i in range(iris.data.shape[0])]

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "onehot.pmml"
        model = LinearRegression()
        pipeline_obj = Pipeline([
            ("mapper", DataFrameMapper([
                (["new"], OneHotEncoder(categories='auto'))
            ])),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile("onehot.pmml"), True)

        # 2
        reg_tab = pmml_obj.RegressionModel[0].RegressionTable[0]
        self.assertEqual(reg_tab.intercept, model.intercept_)

        # 3
        for model_val, pmml_val in zip(model.coef_, reg_tab.NumericPredictor):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

    def test_sklearn_38(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['origin'], axis=1)
        y = df['origin']

        features = [name for name in df.columns if name not in ('origin')]
        target = 'origin'
        f_name = "countvec.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', CountVectorizer())
            ])),
            ('model', model)
        ])

        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_39(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        irisd['new'] = [i % 3 for i in range(iris.data.shape[0])]

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "polyfeat.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ("mapper", DataFrameMapper([
                (["new"], PolynomialFeatures())
            ])),
            ("model", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_40(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        target = 'Species'
        features = irisd.columns.drop('Species')
        file_name = 'mlp_model_numlti_class_classification.pmml'
        model = MLPClassifier()
        pipe = Pipeline([
            ('lag', Lag(aggregation="stddev", value=3)),
            ('model', model)
        ])

        pipe.fit(irisd[features], irisd[target])
        skl_to_pmml(pipe, iris.feature_names, target, file_name)
        pmml_obj = pml.parse(file_name, True)

        # 1
        self.assertEqual(os.path.isfile(file_name), True)

        # 2
        self.assertEqual(2, pmml_obj.NeuralNetwork[0].NeuralLayer.__len__())

        # 3

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SOFTMAX.value,
                         pmml_obj.NeuralNetwork[0].NeuralLayer[1].normalizationMethod)

        # 4

        a_fn = NN_ACTIVATION_FUNCTION.RECTIFIER.value
        self.assertEqual(a_fn, pmml_obj.NeuralNetwork[0].activationFunction)

        # 5
        self.assertEqual(4, pmml_obj.NeuralNetwork[0].NeuralInputs.numberOfInputs)

        # 6
        for model_val, pmml_val in zip(model.intercepts_[0], pmml_obj.NeuralNetwork[0].NeuralLayer[0].Neuron):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.bias))

    def test_sklearn_41(self):
        iris = datasets.load_iris()
        file_name = 'kmeans_model.pmml'
        model = KMeans()
        pipe = Pipeline([
            ('model', model)
        ])

        pipe.fit(iris.data)
        skl_to_pmml(pipe, iris.feature_names, 'target', file_name)
        pmml_obj = pml.parse(file_name, True)

        # 1
        self.assertEqual(os.path.isfile(file_name), True)

        # 2
        self.assertEqual(model.n_clusters, pmml_obj.ClusteringModel[0].Cluster.__len__())

        # 4
        model_clusters = model.cluster_centers_
        clusters_tab = pmml_obj.ClusteringModel[0]
        for mod_cluster, cluster in zip(model_clusters, clusters_tab.Cluster):
            cluster_val = cluster.Array.get_valueOf_()
            cluster_splits = cluster_val.split()
            for model_val, pmml_val in zip(mod_cluster, cluster_splits):
                self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(float(pmml_val)))

    def test_sklearn_42(self):
        titanic = pd.read_csv("nyoka/tests/titanic_train.csv")
        features = titanic.columns
        target = 'Survived'
        f_name = "gb_pmml.pmml"
        model = GradientBoostingClassifier(n_estimators=10)
        pipeline_obj = Pipeline([
            ('scaler', MaxAbsScaler()),
            ('model', model)
        ])

        pipeline_obj.fit(titanic[features], titanic[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 2
        segments = pmml_obj.MiningModel[0].Segmentation.Segment[0].MiningModel.Segmentation.Segment
        estms = model.estimators_.ravel()
        self.assertEqual(len(segments), len(estms))
        for segment, estm in zip(segments, estms):
            values = []
            scores = []
            for nd in segment.TreeModel.Node.Node:
                self.parse_nodes(nd, values, scores)
            values.append(-2)
            scores.insert(0, -2)
            for a, b in zip(scores, estm.tree_.value.ravel()):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
            for a, b in zip(values, estm.tree_.threshold):
                if a == -2:
                    continue
                self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

        # 4
        self.assertEqual(model.min_samples_split, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 5

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[1].RegressionModel.normalizationMethod)

    def test_sklearn_43(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "gb_pmml_iris.pmml"
        model = GradientBoostingClassifier(n_estimators=100)
        pipeline_obj = Pipeline([
            ("imp", Imputer(strategy="median")),
            ("gbc", model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 4
        segments_ = pmml_obj.MiningModel[0].Segmentation.Segment[:-1]
        estms_ = model.estimators_
        for i in range(3):
            segments = segments_[i].MiningModel.Segmentation.Segment
            estms = estms_[:, i]
            self.assertEqual(len(segments), len(estms))
            for segment, estm in zip(segments, estms):
                values = []
                scores = []
                for nd in segment.TreeModel.Node.Node:
                    self.parse_nodes(nd, values, scores)
                values.append(-2)
                scores.insert(0, -2)
                for a, b in zip(scores, estm.tree_.value.ravel()):
                    if a == -2:
                        continue
                    self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))
                for a, b in zip(values, estm.tree_.threshold):
                    if a == -2:
                        continue
                    self.assertEqual("{:.12f}".format(float(a)), "{:.12f}".format(b))

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value,
                         pmml_obj.MiningModel[0].Segmentation.multipleModelMethod)

        # 5
        self.assertEqual(4, pmml_obj.MiningModel[0].Segmentation.Segment.__len__())

        # 6

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SOFTMAX.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[-1].RegressionModel.normalizationMethod)

    def test_sklearn_44(self):
        from sklearn.neural_network import MLPClassifier
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['target'] = [i % 2 for i in range(iris.data.shape[0])]
        target = 'target'
        features = irisd.columns.drop('target')
        file_name = 'mlp_model_binary_class_classification.pmml'
        model = MLPClassifier()
        pipe = Pipeline([
            ('lag', Lag(aggregation="sum", value=3)),
            ('model', model)
        ])

        pipe.fit(irisd[features], irisd[target])
        skl_to_pmml(pipe, iris.feature_names, target, file_name)
        pmml_obj = pml.parse(file_name, True)

        # 1
        self.assertEqual(os.path.isfile(file_name), True)

        # 2
        self.assertEqual(4, pmml_obj.NeuralNetwork[0].NeuralLayer.__len__())

        # 3

        self.assertEqual(NN_ACTIVATION_FUNCTION.IDENTITY.value,
                         pmml_obj.NeuralNetwork[0].NeuralLayer[1].activationFunction)

        # 4

        a_fn = NN_ACTIVATION_FUNCTION.RECTIFIER.value
        self.assertEqual(a_fn, pmml_obj.NeuralNetwork[0].activationFunction)

        # 5
        self.assertEqual(4, pmml_obj.NeuralNetwork[0].NeuralInputs.numberOfInputs)

        # 6
        for model_val, pmml_val in zip(model.intercepts_[0], pmml_obj.NeuralNetwork[0].NeuralLayer[0].Neuron):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.bias))

    def test_sklearn_45(self):
        irisdata = datasets.load_iris()
        iris = pd.DataFrame(irisdata.data, columns=irisdata.feature_names)
        iris['Species'] = irisdata.target

        feature_names = iris.columns.drop('Species')

        X = iris[iris.columns.drop(['Species'])]
        model = IsolationForest()
        pipeline_obj = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('Imputer', Imputer()),
            ('model', model)
        ])

        pipeline_obj.fit(X)
        skl_to_pmml(pipeline_obj, feature_names, pmml_f_name="iforest.pmml")
        pmml_obj = pml.parse("iforest.pmml", True)

        seg_tab = pmml_obj.AnomalyDetectionModel[0].MiningModel.Segmentation.Segment

        pmml_record_count_list = []
        model_record_count_list = []
        pmml_value_list = []
        model_value_list = []
        pmml_score_list = []

        for estimators_tab, dtreg_tab in zip(model.estimators_, seg_tab):
            record_count_val = estimators_tab.tree_.n_node_samples
            value = estimators_tab.tree_.threshold
            for model_record_count, model_record_val in zip(record_count_val, value):
                model_record_count_list.append(model_record_count)
                model_value_list.append(model_record_val)

            count = dtreg_tab.TreeModel.Node.recordCount
            pmml_record_count_list.append(count)

            node_tab = dtreg_tab.TreeModel.Node.Node
            for node in node_tab:
                varlen = node.get_Node().__len__()
                if varlen > 0:
                    pmml_record_count_list.append(node.recordCount)
                    pmml_value_list.append(node.SimplePredicate.value)
                    self.extractValues(node, pmml_record_count_list, pmml_value_list, pmml_score_list)
                else:
                    pmml_record_count_list.append(node.recordCount)
                    pmml_value_list.append(node.SimplePredicate.value)
                    pmml_score_list.append(node.score)

            # 1
            temp = []
            for model_val, pmml_val in zip(value, pmml_value_list):
                model_val_str = str(model_val)
                if model_val_str == "-2.0":
                    temp_len = len(temp) - 1
                    self.assertEqual(temp[temp_len], pmml_val)
                    temp.pop(temp_len)
                else:
                    temp.append(model_val_str)
                    self.assertEqual(model_val_str, pmml_val)
            pmml_value_list.clear()
            model_value_list.clear()

        # 2


        # 3
        self.assertEqual(os.path.isfile("iforest.pmml"), True)

        # 4
        self.assertEqual(model.n_estimators,
                         pmml_obj.AnomalyDetectionModel[0].MiningModel.Segmentation.Segment.__len__())

        # 5
        self.assertEqual(pmml_obj.AnomalyDetectionModel[0].MiningModel.Segmentation.multipleModelMethod, "average")

    def test_sklearn_46(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['new'] = [i % 2 for i in range(irisd.shape[0])]
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "lb_two.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('new', LabelBinarizer())
            ])),
            ('model', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]), \
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_47(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        target = 'Species'
        features = irisd.columns.drop('Species')
        f_name = "imputer.pmml"

        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ('new', StandardScaler()),
            ('imputer', Imputer()),
            ('model', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        segmentation = pmml_obj.MiningModel[0].Segmentation

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(model.classes_.__len__() + 1, segmentation.Segment.__len__())

        # 3

        self.assertEqual(MULTIPLE_MODEL_METHOD.MODEL_CHAIN.value, segmentation.multipleModelMethod)

        # 4

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX.value,
                         segmentation.Segment[-1].RegressionModel.normalizationMethod)

        # 5
        for i in range(model.classes_.__len__()):
            self.assertEqual("{:.12f}".format(model.intercept_[i]),
                             "{:.12f}".format(segmentation.Segment[i].RegressionModel.RegressionTable[0].intercept))

        # 6
        for model_coef, pmml_seg in zip(model.coef_, segmentation.Segment):
            if int(pmml_seg.id) < 4:
                num_predict = pmml_seg.RegressionModel.RegressionTable[0].NumericPredictor
                for model_val, pmml_val in zip(model_coef, num_predict):
                    self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.coefficient))

        # 7

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.LOGISTIC.value,
                         pmml_obj.MiningModel[0].Segmentation.Segment[
                             1].RegressionModel.normalizationMethod)

    def test_sklearn_48(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        target = 'Species'
        features = irisd.columns.drop('Species')
        f_name = "kbins.pmml"
        model = LogisticRegression()
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                (['sepal length (cm)'], KBinsDiscretizer()),
            ])),
            ('model', model)
        ])
        pipeline_obj.fit(irisd[features], irisd[target])
        with self.assertRaises(TypeError):
            skl_to_pmml(pipeline_obj, features, target, f_name)

    def test_sklearn_49(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        target = 'Species'
        features = irisd.columns.drop('Species')
        f_name = "gpc.pmml"
        model = GaussianProcessClassifier()
        pipeline_obj = Pipeline([
            ('model', model)
        ])
        pipeline_obj.fit(irisd[features], irisd[target])
        with self.assertRaises(NotImplementedError):
            skl_to_pmml(pipeline_obj, numpy.array(features), target, f_name)

    def test_sklearn_50(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        target = 'Species'
        features = irisd.columns.drop('Species')
        f_name = "no_pipeline.pmml"
        model = GaussianProcessClassifier()
        model.fit(irisd[features], irisd[target])
        with self.assertRaises(TypeError):
            skl_to_pmml(model, features, target, f_name)

    def test_sklearn_51(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_linear.pmml"
        model = SVC(kernel='linear')
        pipeline_obj = Pipeline([
            ('svm', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2

        self.assertEqual(SVM_CLASSIFICATION_METHOD.OVO.value,
                         pmml_obj.SupportVectorMachineModel[0].classificationMethod)

        # 3
        svm_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine
        for model_val, pmml_val in zip(model.intercept_, svm_tab):
            self.assertEqual(model_val, pmml_val.Coefficients.absoluteValue)

        # 4
        for model_vectors, pmml_vertors in zip(model.support_vectors_,
                                               pmml_obj.SupportVectorMachineModel[0].VectorDictionary.VectorInstance):
            entries_val = pmml_vertors.REAL_SparseArray.REAL_Entries
            arr = numpy.array(entries_val, dtype=numpy.float64)
            for model_val, pmml_val in zip(model_vectors, arr):
                self.assertEqual(model_val, pmml_val)

        # 5
        sv_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine[0]
        for model_val, pmml_val in zip(model.dual_coef_[0], sv_tab.Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_52(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_poly.pmml"
        model = SVC(kernel='poly')
        pipeline_obj = Pipeline([
            ('svm', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        svm_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(pmml_obj.SupportVectorMachineModel[0].classificationMethod, "OneAgainstOne")

        # 3
        for model_val, pmml_val in zip(model.intercept_, svm_tab):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.Coefficients.absoluteValue))

        # 4
        for model_vectors, pmml_vertors in zip(model.support_vectors_,
                                               pmml_obj.SupportVectorMachineModel[0].VectorDictionary.VectorInstance):
            entries_val = pmml_vertors.REAL_SparseArray.REAL_Entries
            arr = numpy.array(entries_val, dtype=numpy.float64)
            for model_val, pmml_val in zip(model_vectors, arr):
                self.assertEqual(model_val, pmml_val)

        # 5
        sv_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine[0]
        for model_val, pmml_val in zip(model.dual_coef_[0], sv_tab.Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_53(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_poly.pmml"
        model = SVC(kernel='sigmoid')
        pipeline_obj = Pipeline([
            ('svm', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        svm_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2

        self.assertEqual(SVM_CLASSIFICATION_METHOD.OVO.value,
                         pmml_obj.SupportVectorMachineModel[0].classificationMethod)

        # 3
        for model_val, pmml_val in zip(model.intercept_, svm_tab):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.Coefficients.absoluteValue))

        # 4
        for model_vectors, pmml_vertors in zip(model.support_vectors_,
                                               pmml_obj.SupportVectorMachineModel[0].VectorDictionary.VectorInstance):
            entries_val = pmml_vertors.REAL_SparseArray.REAL_Entries
            arr = numpy.array(entries_val)
            arr = numpy.array(entries_val, dtype=numpy.float64)
            for model_val, pmml_val in zip(model_vectors, arr):
                self.assertEqual(model_val, pmml_val)

        # 5
        sv_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine[0]
        for model_val, pmml_val in zip(model.dual_coef_[0], sv_tab.Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_54(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = [i % 2 for i in range(irisd.shape[0])]

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_bin.pmml"
        model = SVC()
        pipeline_obj = Pipeline([
            ('svm', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2

        self.assertEqual(SVM_CLASSIFICATION_METHOD.OVO.value,
                         pmml_obj.SupportVectorMachineModel[0].classificationMethod)

        # 3
        svm_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine
        for model_val, pmml_val in zip(model.intercept_, svm_tab):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.Coefficients.absoluteValue))

        # 4
        vect_tab = pmml_obj.SupportVectorMachineModel[0].VectorDictionary.VectorInstance
        for model_vectors, pmml_vertors in zip(model.support_vectors_, vect_tab):
            entries_val = pmml_vertors.REAL_SparseArray.REAL_Entries
            arr = numpy.array(entries_val)
            arr = numpy.array(entries_val, dtype=numpy.float64)
            for model_val, pmml_val in zip(model_vectors, arr):
                self.assertEqual(model_val, pmml_val)

        # 5
        sv_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine[0]
        for model_val, pmml_val in zip(model.dual_coef_[0], sv_tab.Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_55(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = [i % 2 for i in range(irisd.shape[0])]

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_bin.pmml"
        model = SVC()
        pipeline_obj = Pipeline([
            ('svm', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target].astype(float))
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(pmml_obj.SupportVectorMachineModel[0].classificationMethod, "OneAgainstOne")

        # 3
        svm_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine
        for model_val, pmml_val in zip(model.intercept_, svm_tab):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.Coefficients.absoluteValue))

        # 4
        vect_tab = pmml_obj.SupportVectorMachineModel[0].VectorDictionary.VectorInstance
        for model_vectors, pmml_vertors in zip(model.support_vectors_, vect_tab):
            entries_val = pmml_vertors.REAL_SparseArray.REAL_Entries
            arr = numpy.array(entries_val)
            arr = numpy.array(entries_val, dtype=numpy.float64)
            for model_val, pmml_val in zip(model_vectors, arr):
                self.assertEqual(model_val, pmml_val)

        # 5
        sv_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine[0]
        for model_val, pmml_val in zip(model.dual_coef_[0], sv_tab.Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_56(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = [i % 2 for i in range(irisd.shape[0])]

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "svc_bin.pmml"
        model = SVC()
        pipeline_obj = Pipeline([
            ('svm', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target].astype(str))
        skl_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(pmml_obj.SupportVectorMachineModel[0].classificationMethod, "OneAgainstOne")

        # 3
        svm_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine
        for model_val, pmml_val in zip(model.intercept_, svm_tab):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.Coefficients.absoluteValue))

        # 4
        vect_tab = pmml_obj.SupportVectorMachineModel[0].VectorDictionary.VectorInstance
        for model_vectors, pmml_vertors in zip(model.support_vectors_, vect_tab):
            entries_val = pmml_vertors.REAL_SparseArray.REAL_Entries
            arr = numpy.array(entries_val)
            arr = numpy.array(entries_val, dtype=numpy.float64)
            for model_val, pmml_val in zip(model_vectors, arr):
                self.assertEqual(model_val, pmml_val)

        # 5
        sv_tab = pmml_obj.SupportVectorMachineModel[0].SupportVectorMachine[0]
        for model_val, pmml_val in zip(model.dual_coef_[0], sv_tab.Coefficients.Coefficient):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.value))

    def test_sklearn_57(self):
        df = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = df.drop(['mpg'], axis=1)
        y = df['mpg']
        features = [name for name in df.columns if name not in ('mpg')]
        target = 'mpg'
        f_name = "mlpr_pmml.pmml"
        model = MLPRegressor()
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', TfidfVectorizer())
            ])),
            ('model', model)
        ])
        pipeline_obj.fit(X, y)
        skl_to_pmml(pipeline_obj, features, target, f_name)

        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(2, pmml_obj.NeuralNetwork[0].NeuralLayer.__len__())

        # 3

        self.assertEqual(NN_ACTIVATION_FUNCTION.RECTIFIER.value, pmml_obj.NeuralNetwork[0].activationFunction)

        # 4
        self.assertEqual(300, pmml_obj.NeuralNetwork[0].NeuralInputs.numberOfInputs)

        for model_val, pmml_val in zip(model.intercepts_[0], pmml_obj.NeuralNetwork[0].NeuralLayer[0].Neuron):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.bias))

    def test_sklearn_58(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target
        irisd['new'] = [i % 2 for i in range(iris.data.shape[0])]

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "mlp2.pmml"
        model = MLPClassifier()
        pipeline_obj = Pipeline([
            ("model", model)
        ])
        pipeline_obj.fit(irisd[features], irisd[target])
        skl_to_pmml(pipeline_obj, features, target, f_name)

        pmml_obj = pml.parse(f_name, True)

        # 1
        self.assertEqual(os.path.isfile(f_name), True)

        # 2
        self.assertEqual(2, pmml_obj.NeuralNetwork[0].NeuralLayer.__len__())

        # 3

        self.assertEqual(REGRESSION_NORMALIZATION_METHOD.SOFTMAX.value,
                         pmml_obj.NeuralNetwork[0].NeuralLayer[1].normalizationMethod)

        # 4

        a_fn = NN_ACTIVATION_FUNCTION.RECTIFIER.value
        self.assertEqual(a_fn, pmml_obj.NeuralNetwork[0].activationFunction)

        # 5
        self.assertEqual(5, pmml_obj.NeuralNetwork[0].NeuralInputs.numberOfInputs)

        # 6
        for model_val, pmml_val in zip(model.intercepts_[0], pmml_obj.NeuralNetwork[0].NeuralLayer[0].Neuron):
            self.assertEqual("{:.12f}".format(model_val), "{:.12f}".format(pmml_val.bias))

    def extractValues(self, node, pmml_record_count_list, pmml_value_list, pmml_score_list):
        for nsample in (node.Node):
            varlen = nsample.get_Node().__len__()
            if varlen > 0:
                pmml_record_count_list.append(nsample.recordCount)
                pmml_value_list.append(nsample.SimplePredicate.value)
                self.extractValues(nsample, pmml_record_count_list, pmml_value_list, pmml_score_list)
            else:
                pmml_record_count_list.append(nsample.recordCount)
                pmml_value_list.append(nsample.SimplePredicate.value)
                pmml_score_list.append(nsample.score)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
