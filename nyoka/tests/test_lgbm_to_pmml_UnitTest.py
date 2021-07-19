import sys, os

import unittest
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
from nyoka import lgb_to_pmml
from nyoka import PMML44 as pml
import json


class TestMethods(unittest.TestCase):

    def test_lgbm_01(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species').to_numpy()
        target = 'Species'
        f_name = "lgbmc_pmml.pmml"
        model = LGBMClassifier()
        pipeline_obj = Pipeline([
            ('lgbmc', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        lgb_to_pmml(pipeline_obj, features, target, f_name, model_name="MyLGBM",
                    description="A Model for test")

        pmml_obj = pml.parse(f_name, True)
        pmml_value_list = []
        model_value_list = []

        pmml_score_list = []
        model_score_list = []

        list_seg_score1 = []
        list_seg_score2 = []
        list_seg_score3 = []

        list_seg_val1 = []
        list_seg_val2 = []
        list_seg_val3 = []

        seg_tab = pmml_obj.MiningModel[0].Segmentation.Segment
        for seg in seg_tab:
            if int(seg.id) <= 3:
                for segment in seg.MiningModel.Segmentation.Segment:
                    node_tab = segment.TreeModel.Node.Node
                    if not node_tab:
                        pmml_score_list.append(segment.TreeModel.Node.score)
                    else:
                        for node in node_tab:
                            varlen = node.get_Node().__len__()
                            if varlen > 0:
                                pmml_value_list.append(node.SimplePredicate.value)
                                self.extractValues(node, pmml_value_list, pmml_score_list)
                            else:
                                pmml_value_list.append(node.SimplePredicate.value)
                                pmml_score_list.append(node.score)

        main_key_value = []
        lgb_dump = model.booster_.dump_model()
        for i in range(len(lgb_dump['tree_info'])):
            tree = lgb_dump['tree_info'][i]['tree_structure']
            main_key_value.append(tree)

        n = 1
        for i in range(len(main_key_value)):
            list_score_temp = []
            list_val_temp = []
            node_list = main_key_value[i]
            if (n == 1):
                n = 2
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score1 = list_seg_score1 + list_score_temp
                list_seg_val1 = list_seg_val1 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif (n == 2):
                n = 3
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score2 = list_seg_score2 + list_score_temp
                list_seg_val2 = list_seg_val2 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif (n == 3):
                n = 1
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score3 = list_seg_score3 + list_score_temp
                list_seg_val3 = list_seg_val3 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()

        model_score_list = list_seg_score1 + list_seg_score2 + list_seg_score3
        model_value_list = list_seg_val1 + list_seg_val2 + list_seg_val3

        ##1
        for model_val, pmml_val in zip(model_score_list, pmml_score_list):
            self.assertEqual(model_val, float(pmml_val))

        ##2
        for model_val, pmml_val in zip(model_value_list, pmml_value_list):
            self.assertEqual(model_val, pmml_val)

        ##3
        self.assertEqual(os.path.isfile(f_name), True)

    def test_lgbm_02(self):

        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        feature_names = [name for name in auto.columns if name not in ('mpg', 'car name')]
        target_name = 'mpg'
        f_name = "lgbmr_pmml.pmml"
        model = LGBMRegressor()
        pipeline_obj = Pipeline([
            ('lgbmr', model)
        ])

        pipeline_obj.fit(auto[feature_names], auto[target_name])

        lgb_to_pmml(pipeline_obj, feature_names, target_name, f_name)

        pmml_obj = pml.parse(f_name, True)

        pmml_value_list = []
        model_value_list = []

        pmml_score_list = []
        model_score_list = []

        seg_tab = pmml_obj.MiningModel[0].Segmentation.Segment
        for seg in seg_tab:
            for node in seg.TreeModel.Node.Node:
                varlen = node.get_Node().__len__()
                if varlen > 0:
                    pmml_value_list.append(node.SimplePredicate.value)
                    self.extractValues(node, pmml_value_list, pmml_score_list)
                else:
                    pmml_value_list.append(node.SimplePredicate.value)
                    pmml_score_list.append(node.score)

        main_key_value = []
        lgb_dump = model.booster_.dump_model()
        for i in range(len(lgb_dump['tree_info'])):
            tree = lgb_dump['tree_info'][i]['tree_structure']
            main_key_value.append(tree)

        for i in range(len(main_key_value)):
            list_score_temp = []
            list_val_temp = []
            node_list = main_key_value[i]
            self.create_node(node_list, list_score_temp, list_val_temp)
            model_score_list = model_score_list + list_score_temp
            model_value_list = model_value_list + list_val_temp
            list_val_temp.clear()
            list_score_temp.clear()

        ##1
        for model_val, pmml_val in zip(model_score_list, pmml_score_list):
            self.assertEqual(model_val, float(pmml_val))

        ##2
        for model_val, pmml_val in zip(model_value_list, pmml_value_list):
            self.assertEqual(model_val, pmml_val)

        ##3
        self.assertEqual(os.path.isfile(f_name), True)

    def test_lgbm_03(self):

        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "lgbmc_pmml_preprocess.pmml"
        model = LGBMClassifier(n_estimators=5)

        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('LGBMC', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        lgb_to_pmml(pipeline_obj, features, target, f_name)
        pmml_obj = pml.parse(f_name, True)

        pmml_value_list = []
        model_value_list = []

        pmml_score_list = []
        model_score_list = []

        list_seg_score1 = []
        list_seg_score2 = []
        list_seg_score3 = []

        list_seg_val1 = []
        list_seg_val2 = []
        list_seg_val3 = []

        seg_tab = pmml_obj.MiningModel[0].Segmentation.Segment
        for seg in seg_tab:
            if int(seg.id) <= 3:
                for segment in seg.MiningModel.Segmentation.Segment:
                    node_tab = segment.TreeModel.Node.Node
                    if not node_tab:
                        pmml_score_list.append(segment.TreeModel.Node.score)
                    else:
                        for node in node_tab:
                            varlen = node.get_Node().__len__()
                            if varlen > 0:
                                pmml_value_list.append(node.SimplePredicate.value)
                                self.extractValues(node, pmml_value_list, pmml_score_list)
                            else:
                                pmml_value_list.append(node.SimplePredicate.value)
                                pmml_score_list.append(node.score)

        main_key_value = []
        lgb_dump = model.booster_.dump_model()
        for i in range(len(lgb_dump['tree_info'])):
            tree = lgb_dump['tree_info'][i]['tree_structure']
            main_key_value.append(tree)

        n = 1
        for i in range(len(main_key_value)):
            list_score_temp = []
            list_val_temp = []
            node_list = main_key_value[i]
            if (n == 1):
                n = 2
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score1 = list_seg_score1 + list_score_temp
                list_seg_val1 = list_seg_val1 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif (n == 2):
                n = 3
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score2 = list_seg_score2 + list_score_temp
                list_seg_val2 = list_seg_val2 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif (n == 3):
                n = 1
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score3 = list_seg_score3 + list_score_temp
                list_seg_val3 = list_seg_val3 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()

        model_score_list = list_seg_score1 + list_seg_score2 + list_seg_score3
        model_value_list = list_seg_val1 + list_seg_val2 + list_seg_val3

        ##1
        for model_val, pmml_val in zip(model_score_list, pmml_score_list):
            self.assertEqual(model_val, float(pmml_val))

        ##2
        for model_val, pmml_val in zip(model_value_list, pmml_value_list):
            self.assertEqual(model_val, pmml_val)

        ##3
        self.assertEqual(os.path.isfile(f_name), True)

    def test_lgbm_04(self):

        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in ('mpg')]

        target_name = 'mpg'
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        f_name = "lgbmr_pmml_preprocess2.pmml"
        model = LGBMRegressor()
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', CountVectorizer()),
                (['displacement'], [StandardScaler()])
            ])),
            ('lgbmr', model)
        ])
        pipeline_obj.fit(x_train, y_train)

        lgb_to_pmml(pipeline_obj, feature_names, target_name, f_name)

        pmml_obj = pml.parse(f_name, True)

        pmml_value_list = []
        model_value_list = []

        pmml_score_list = []
        model_score_list = []

        seg_tab = pmml_obj.MiningModel[0].Segmentation.Segment
        for seg in seg_tab:
            for node in seg.TreeModel.Node.Node:
                varlen = node.get_Node().__len__()
                if varlen > 0:
                    pmml_value_list.append(node.SimplePredicate.value)
                    self.extractValues(node, pmml_value_list, pmml_score_list)
                else:
                    pmml_value_list.append(node.SimplePredicate.value)
                    pmml_score_list.append(node.score)

        main_key_value = []
        lgb_dump = model.booster_.dump_model()
        for i in range(len(lgb_dump['tree_info'])):
            tree = lgb_dump['tree_info'][i]['tree_structure']
            main_key_value.append(tree)

        for i in range(len(main_key_value)):
            list_score_temp = []
            list_val_temp = []
            node_list = main_key_value[i]
            self.create_node(node_list, list_score_temp, list_val_temp)
            model_score_list = model_score_list + list_score_temp
            model_value_list = model_value_list + list_val_temp
            list_val_temp.clear()
            list_score_temp.clear()

        ##1
        for model_val, pmml_val in zip(model_score_list, pmml_score_list):
            self.assertEqual(model_val, float(pmml_val))

        ##2
        for model_val, pmml_val in zip(model_value_list, pmml_value_list):
            self.assertEqual(model_val, pmml_val)

        ##3
        self.assertEqual(os.path.isfile(f_name), True)

    def test_lgbm_05(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['target'] = [i % 2 for i in range(iris.data.shape[0])]

        features = irisd.columns.drop('target')
        target = 'target'
        f_name = "lgbc_bin_pmml.pmml"
        model = LGBMClassifier()
        pipeline_obj = Pipeline([
            ('lgbmc', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])

        lgb_to_pmml(pipeline_obj, features, target, f_name)

        # self.assertEqual(os.path.isfile("lgbc_bin_pmml.pmml"), True)

        pmml_obj = pml.parse(f_name, True)

        pmml_value_list = []
        model_value_list = []

        pmml_score_list = []
        model_score_list = []

        seg_tab = pmml_obj.MiningModel[0].Segmentation.Segment
        for seg in seg_tab:
            if int(seg.id) == 1:
                for segment in seg.MiningModel.Segmentation.Segment:
                    node_tab = segment.TreeModel.Node.Node
                    if not node_tab:
                        pmml_score_list.append(segment.TreeModel.Node.score)
                    else:
                        for node in node_tab:
                            varlen = node.get_Node().__len__()
                            if varlen > 0:
                                pmml_value_list.append(node.SimplePredicate.value)
                                self.extractValues(node, pmml_value_list, pmml_score_list)
                            else:
                                pmml_value_list.append(node.SimplePredicate.value)
                                pmml_score_list.append(node.score)

        main_key_value = []
        lgb_dump = model.booster_.dump_model()
        for i in range(len(lgb_dump['tree_info'])):
            tree = lgb_dump['tree_info'][i]['tree_structure']
            main_key_value.append(tree)

        for i in range(len(main_key_value)):
            list_score_temp = []
            list_val_temp = []
            node_list = main_key_value[i]
            self.create_node(node_list, list_score_temp, list_val_temp)
            model_score_list = model_score_list + list_score_temp
            model_value_list = model_value_list + list_val_temp
            list_val_temp.clear()
            list_score_temp.clear()

        ##1
        for model_val, pmml_val in zip(model_score_list, pmml_score_list):
            self.assertEqual(model_val, float(pmml_val))

        ##2
        for model_val, pmml_val in zip(model_value_list, pmml_value_list):
            self.assertEqual(model_val, pmml_val)

        ##3
        self.assertEqual(os.path.isfile(f_name), True)

    def test_lgbm_06(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['target'] = [i % 2 for i in range(iris.data.shape[0])]

        features = irisd.columns.drop('target')
        target = 'target'

        model = LGBMClassifier()

        model.fit(irisd[features], irisd[target])

        with self.assertRaises(TypeError):
            lgb_to_pmml(model, features, target, "lgbc_bin_pmml.pmml")

    def test_lgbm_07(self):
        iris = datasets.load_iris()
        abc = ['f1', 'f2', 'f3', 'f4']
        irisd = pd.DataFrame(iris.data, columns=abc)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "lgbmc_pmml_with_f_column_names.pmml"
        pipeline_obj = Pipeline([
            ('scaler', StandardScaler()),
            ('lgbmc', LGBMClassifier())
        ])
        pipeline_obj.fit(irisd[features].values, irisd[target].values)
        lgb_to_pmml(pipeline_obj, abc, target, f_name, model_name="MyLGBM")

    def extractValues(self, node, pmml_value_list, pmml_score_list):
        for nsample in (node.Node):
            varlen = nsample.get_Node().__len__()
            if varlen > 0:
                pmml_value_list.append(nsample.SimplePredicate.value)
                self.extractValues(nsample, pmml_value_list, pmml_score_list)
            else:
                pmml_value_list.append(nsample.SimplePredicate.value)
                pmml_score_list.append(nsample.score)

    def create_node(self, obj, list_score_temp, list_val_temp):
        if 'leaf_index' in obj:
            list_score_temp.append(obj['leaf_value'])
        else:
            self.create_left_node(obj, list_score_temp, list_val_temp)
            self.create_right_node(obj, list_score_temp, list_val_temp)

    def create_left_node(self, obj, list_score_temp, list_val_temp):
        value = "{:.16f}".format(obj['threshold'])
        list_val_temp.append(value)
        self.create_node(obj['left_child'], list_score_temp, list_val_temp)

    def create_right_node(self, obj, list_score_temp, list_val_temp):
        value = "{:.16f}".format(obj['threshold'])
        list_val_temp.append(value)
        self.create_node(obj['right_child'], list_score_temp, list_val_temp)


if __name__ == '__main__':
    unittest.main(warnings='ignore')







