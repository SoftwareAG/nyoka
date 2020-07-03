import sys, os

import unittest
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from nyoka import xgboost_to_pmml
from nyoka import PMML44 as pml
import json


class TestMethods(unittest.TestCase):

    def test_xgboost_01(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species').to_numpy()
        target = 'Species'
        f_name = "xgbc_pmml.pmml"
        model = XGBClassifier()
        pipeline_obj = Pipeline([
            ('xgbc', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        xgboost_to_pmml(pipeline_obj, features, target, f_name, model_name="testModel")
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

        get_nodes_in_json_format = []
        for i in range(model.n_estimators * model.n_classes_):
            get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))

        n = 1
        for i in range(len(get_nodes_in_json_format)):
            list_score_temp = []
            list_val_temp = []
            node_list = get_nodes_in_json_format[i]
            if n == 1:
                n = 2
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score1 = list_seg_score1 + list_score_temp
                list_seg_val1 = list_seg_val1 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif n == 2:
                n = 3
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score2 = list_seg_score2 + list_score_temp
                list_seg_val2 = list_seg_val2 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif n == 3:
                n = 1
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score3 = list_seg_score3 + list_score_temp
                list_seg_val3 = list_seg_val3 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()

        model_score_list = list_seg_score1 + list_seg_score2 + list_seg_score3
        model_value_list = list_seg_val1 + list_seg_val2 + list_seg_val3

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

        ##1
        for model_val, pmml_val in zip(model_score_list, pmml_score_list):
            self.assertEqual(model_val, float(pmml_val))

        ##2
        for model_val, pmml_val in zip(model_value_list, pmml_value_list):
            self.assertEqual(model_val, pmml_val)

        ##3
        self.assertEqual(os.path.isfile(f_name), True)

    def test_xgboost_02(self):
        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        feature_names = [name for name in auto.columns if name not in ('mpg', 'car name')]
        target_name = 'mpg'
        f_name = "xgbr_pmml.pmml"
        model = XGBRegressor()
        pipeline_obj = Pipeline([
            ('xgbr', model)
        ])

        pipeline_obj.fit(auto[feature_names], auto[target_name])
        xgboost_to_pmml(pipeline_obj, feature_names, target_name, f_name, description="A test model")
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

        get_nodes_in_json_format = []
        for i in range(model.n_estimators):
            get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))

        for i in range(len(get_nodes_in_json_format)):
            list_score_temp = []
            list_val_temp = []
            node_list = get_nodes_in_json_format[i]
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

    def test_xgboost_03(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "xgbc_pmml_preprocess.pmml"
        model = XGBClassifier(n_estimators=5)
        pipeline_obj = Pipeline([
            ('scaling', StandardScaler()),
            ('xgbc', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        xgboost_to_pmml(pipeline_obj, features, target, f_name)
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

        get_nodes_in_json_format = []
        for i in range(model.n_estimators * model.n_classes_):
            get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))

        n = 1
        for i in range(len(get_nodes_in_json_format)):
            list_score_temp = []
            list_val_temp = []
            node_list = get_nodes_in_json_format[i]
            if n == 1:
                n = 2
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score1 = list_seg_score1 + list_score_temp
                list_seg_val1 = list_seg_val1 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif n == 2:
                n = 3
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score2 = list_seg_score2 + list_score_temp
                list_seg_val2 = list_seg_val2 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()
            elif n == 3:
                n = 1
                self.create_node(node_list, list_score_temp, list_val_temp)
                list_seg_score3 = list_seg_score3 + list_score_temp
                list_seg_val3 = list_seg_val3 + list_val_temp
                list_val_temp.clear()
                list_score_temp.clear()

        model_score_list = list_seg_score1 + list_seg_score2 + list_seg_score3
        model_value_list = list_seg_val1 + list_seg_val2 + list_seg_val3

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

        ##1
        for model_val, pmml_val in zip(model_score_list, pmml_score_list):
            self.assertEqual(model_val, float(pmml_val))

        ##2
        for model_val, pmml_val in zip(model_value_list, pmml_value_list):
            self.assertEqual(model_val, pmml_val)

        ##3
        self.assertEqual(os.path.isfile(f_name), True)

    def test_xgboost_04(self):
        auto = pd.read_csv('nyoka/tests/auto-mpg.csv')
        X = auto.drop(['mpg'], axis=1)
        y = auto['mpg']

        feature_names = [name for name in auto.columns if name not in 'mpg']
        f_name = "xgbr_pmml_preprocess2.pmml"
        target_name = 'mpg'
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        model = XGBRegressor()
        pipeline_obj = Pipeline([
            ('mapper', DataFrameMapper([
                ('car name', CountVectorizer()),
                (['displacement'], [StandardScaler()])
            ])),
            ('xgbr', model)
        ])

        pipeline_obj.fit(x_train, y_train)
        xgboost_to_pmml(pipeline_obj, feature_names, target_name, f_name)
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

        get_nodes_in_json_format = []
        for i in range(model.n_estimators):
            get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))

        for i in range(len(get_nodes_in_json_format)):
            list_score_temp = []
            list_val_temp = []
            node_list = get_nodes_in_json_format[i]
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

    def test_xgboost_05(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['target'] = [i % 2 for i in range(iris.data.shape[0])]

        features = irisd.columns.drop('target')
        target = 'target'
        f_name = "xgbc_bin_pmml.pmml"
        model = XGBClassifier(min_child_weight=6, n_estimators=10, scale_pos_weight=10, deterministic_histogram=False)
        pipeline_obj = Pipeline([
            ('xgbc', model)
        ])

        pipeline_obj.fit(irisd[features], irisd[target])
        xgboost_to_pmml(pipeline_obj, features, target, f_name)
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

        get_nodes_in_json_format = []
        for i in range(model.n_estimators):
            get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))

        for i in range(len(get_nodes_in_json_format)):
            list_score_temp = []
            list_val_temp = []
            node_list = get_nodes_in_json_format[i]
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



    def test_xgboost_06(self):
        iris = datasets.load_iris()
        irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
        irisd['Species'] = iris.target

        features = irisd.columns.drop('Species')
        target = 'Species'
        f_name = "xgbc_pmml.pmml"

        model = XGBClassifier()

        model.fit(irisd[features], irisd[target])

        with self.assertRaises(TypeError):
            xgboost_to_pmml(model, features, target,f_name , model_name="testModel")

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
        if 'split' not in obj:
            list_score_temp.append(obj['leaf'])
        else:
            self.create_left_node(obj, list_score_temp, list_val_temp)
            self.create_right_node(obj, list_score_temp, list_val_temp)

    def create_left_node(self, children_list, list_score_temp, list_val_temp):
        value = "{:.16f}".format(children_list['split_condition'])
        list_val_temp.append(value)
        self.create_node(children_list['children'][0], list_score_temp, list_val_temp)

    def create_right_node(self, children_list, list_score_temp, list_val_temp):
        value = "{:.16f}".format(children_list['split_condition'])
        list_val_temp.append(value)
        self.create_node(children_list['children'][1], list_score_temp, list_val_temp)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
