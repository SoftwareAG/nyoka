import nyoka.PMML43Ext as pml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor

import sys

class Tree():
    all_node_list = list()
    root = None
    fields = list()
    classes = list()

    class Node():
        def __init__(self):
            self.field = ''
            self.value = -2
            self.score = -2
            self.left = None
            self.right = None
            self.parent = None
            self.operator = None

    def __init__(self, features, classes=None, operator=None):
        self.root = self.Node()
        self.fields = features
        self.classes = classes
        self.all_node_list = list()
        self.operator = operator

    def get_node_info(self, all_node):
        for node in all_node:
            if not node.get_score():
                score = -2
            else:
                score = node.get_score()
            sp = node.get_SimplePredicate()
            value = sp.get_value()
            field = sp.get_field()
            operator = sp.get_operator()
            self.all_node_list.append((score, field, value, operator))
            if len(node.get_Node()) > 0:
                self.get_node_info(node.get_Node())
        self.root.field = self.all_node_list[0][1]
        self.root.value = self.all_node_list[0][2]

    def predict(self, sample):
        prob = list()
        idx = 0
        for rec in sample:
            prob.append(self._predict(self.root, rec, idx))
            idx +=1
        return np.array(prob)

    def _predict(self, root, sample, index):
        if root.value == -2:
            if len(self.classes) > 1:
                prob = [0] * len(self.classes)
                if self.classes[0].__class__.__name__ == 'int':
                    prob[self.classes.index(int(root.score))] = 1
                else:
                    prob[self.classes.index(root.score)] = 1
            else:
                prob = [float(root.score)]
            return prob
        idx = self.fields.index(root.field)
        if self.operator == 'lessThan':
            if sample[idx] < float(root.value):
                result = self._predict(root.left, sample, index)
            else:
                result = self._predict(root.right, sample, index)
        else:
            if sample[idx] <= float(root.value):
                result = self._predict(root.left, sample, index)
            else:
                result = self._predict(root.right, sample, index)

        return result

    def build_tree(self):
        cur_node = self.root
        for nd in self.all_node_list:
            if nd[2] == cur_node.value:
                if nd[-1] == 'lessOrEqual' or nd[-1] == 'lessThan':
                    cur_node.left = self.Node()
                    if nd[0] != -2:
                        cur_node.left.score = nd[0]
                    else:
                        cur_node.left.parent = cur_node
                        cur_node = cur_node.left
                else:
                    cur_node.right = self.Node()
                    if nd[0] != -2:
                        cur_node.right.score = nd[0]
                        cur_node = cur_node.parent
                        while cur_node and cur_node.right:
                            cur_node = cur_node.parent
                            if not cur_node:
                                break
                    else:
                        cur_node.right.parent = cur_node
                        cur_node = cur_node.right
            else:
                cur_node.field = nd[1]
                cur_node.value = nd[2]
                if nd[-1] == 'lessOrEqual' or nd[-1] == 'lessThan':
                    cur_node.left = self.Node()
                    if nd[0] != -2:
                        cur_node.left.score = nd[0]
                    else:
                        cur_node.left.parent = cur_node
                        cur_node = cur_node.left
                else:
                    cur_node.right = self.Node()
                    if nd[0] != -2:
                        cur_node.right.score = nd[0]
                        cur_node = cur_node.parent
                        while cur_node and cur_node.right:
                            cur_node = cur_node.parent
                            if not cur_node:
                                break
                    else:
                        cur_node.right.parent = cur_node
                        cur_node = cur_node.right


def reconstruct(pmml,*args):
    mining_model = pmml.get_MiningModel()[0]
    targets = mining_model.get_Targets()
    model = EnsembleModel()
    model.isRescaleRequired(val=True)
    if targets:
        model.set_target_transformation(targets)
    fields, classes = model.get_data_information(pmml)
    if args:
        fields = args[0]
    segmentations = mining_model.get_Segmentation()
    model.set_multiple_model_method(segmentations.get_multipleModelMethod())
    segments = segmentations.get_Segment()
    tree_models = model.get_tree_models( segments)
    tree_objs = model.get_tree_objects(tree_models, fields, classes)
    model.set_classes(classes)
    model.set_tree_objs(tree_objs)
    extension_value=pmml.get_MiningBuildTask().get_Extension()[0].get_value()
    if 'XGBRegressor' in extension_value:
        mod=XGBRegressor()
        mod.rescaleConstant = model.get_rescale_Constant(targets)
    if 'GradientBoostingRegressor' in extension_value:
        mod=GradientBoostingRegressor()
        mod.rescaleFactor = model.get_rescale_Factor(targets)
        mod.rescaleConstant = model.get_rescale_Constant(targets)
    elif 'RandomForestClassifier' in extension_value:
        mod=RandomForestClassifier()
    elif 'RandomForestRegressor' in extension_value:
        mod=RandomForestRegressor()
    elif 'LGBMRegressor' in extension_value:
        mod=LGBMRegressor()
        model.isRescaleRequired(val=False)
    elif 'LGBMClassifier' in extension_value:
        mod=LGBMClassifier()
    elif 'XGBClassifier' in extension_value:
        mod=XGBClassifier()
    elif 'GradientBoostingClassifier' in extension_value:
        mod=GradientBoostingClassifier()
        mod.normalizationMethod=model.normalizationMethod
    mod.predict = model.predict
    mod.multiple_model_method=model.multiple_model_method
    mod.trees=model.trees
    mod.transformedOutputs=model.transformedOutputs
    mod.classes=model.classes
    return mod




class EnsembleModel:

    def __init__(self):
        self.transformedOutputs = list()

    def set_rescaleFactor(self, rescaleFactor):
        self.rescaleFactor = rescaleFactor

    def set_rescaleConstant(self, rescaleConstant):
        self.rescaleConstant = rescaleConstant

    def set_multiple_model_method(self, method):
        self.multiple_model_method = method

    def set_normalization_method(self, method):
        self.normalizationMethod = method

    def set_target_transformation(self, isPresent):
        self.targetTransformation = isPresent

    def set_classes(self, classes):
        self.classes = classes

    def set_tree_objs(self, tree_objs):
        self.trees = tree_objs

    def set_target_transformation(self, targets):
        target = targets.get_Target()[0]
        self.targetTransformation = True
        self.rescaleFactor = target.get_rescaleFactor()
        self.rescaleConstant = target.get_rescaleConstant()

    def get_rescale_Constant(self,targets):
        target = targets.get_Target()[0]
        return target.get_rescaleConstant()

    def get_rescale_Factor(self,targets):
        target = targets.get_Target()[0]
        return target.get_rescaleFactor()

    def isRescaleRequired(self,val=True):
        self.rescaleRequired=val


    def set_transformed_output_info(self, outputFields):
        transformed = list()
        def get_apply_(apply):
            if type(apply)==list:
                apply = apply[0]
            function = apply.get_function()
            constant = apply.get_Constant()[0].get_valueOf_()
            transformed.append((function, constant))
            if apply.get_Apply():
                get_apply_(apply.get_Apply())
        if len(outputFields) > 1:
            transformedField = outputFields[-1]
            get_apply_(transformedField.get_Apply())
            self.transformedOutputs.append(transformed)


    def get_tree_models_for_modelChain(self, segments):
        tree_models = list()
        for segment in segments:
            if segment.get_MiningModel():
                inner_model = segment.get_MiningModel()
                output = inner_model.get_Output()
                self.set_transformed_output_info(output.get_OutputField())
                inner_segmentation = inner_model.get_Segmentation()
                inner_segments = inner_segmentation.get_Segment()
                tree_models_inner = list()
                for segment in inner_segments:
                    tree_models_inner.append(segment.get_TreeModel())
                tree_models.append(tree_models_inner)
            else:
                reg_model = segment.get_RegressionModel()
                self.set_normalization_method(reg_model.get_normalizationMethod())
        return tree_models


    def get_tree_models(self, segments):
        if self.multiple_model_method == 'modelChain':
            tree_models = self.get_tree_models_for_modelChain(segments)
        else:
            tree_models = list()
            for segment in segments:
                tree_models.append(segment.get_TreeModel())
        return tree_models

    def get_tree_objects(self, tree_models, fields, classes):

        trees = list()
        for i, tree_model in enumerate(tree_models):
            if 'list' in str(type(tree_model)):
                tree_inner = list()
                for tree_mod in tree_model:
                    main_node = tree_mod.get_Node()
                    all_node = main_node.get_Node()
                    operator = all_node[0].get_SimplePredicate().get_operator()
                    tt = Tree(fields, [1], operator)
                    tt.get_node_info(all_node)
                    tt.build_tree()
                    model = DecisionTreeRegressor()
                    model.n_features = len(fields)
                    model.n_features_ = len(fields)
                    model.n_outputs_ = 1
                    model.n_outputs = 1
                    model.classes_ = np.array(classes)
                    model.tree_ = tt
                    tree_inner.append(model)
                trees.append(tree_inner)
            else:
                main_node = tree_model.get_Node()
                all_node = main_node.get_Node()
                operator = all_node[0].get_SimplePredicate().get_operator()
                tt = Tree(fields, classes, operator)
                tt.get_node_info(all_node)
                tt.build_tree()
                model = DecisionTreeClassifier()
                model.n_features = len(fields)
                model.n_features_ = len(fields)
                model.n_outputs_ = 1
                model.n_outputs = 1
                model.classes_ = np.array(classes)
                model._estimator_type = 'classifier' if len(classes) > 0 else 'regressor'
                model.tree_ = tt
                trees.append(model)
        return trees

    def get_data_information(self, pmml):
        tree_model = pmml.get_MiningModel()[0]
        mining = tree_model.get_MiningSchema()
        mfs = mining.get_MiningField()
        fields = list()
        classes = list()
        for mm in mfs:
            if mm.get_usageType() != 'target':
                fields.append(mm.get_name())
            else:
                target_name = mm.get_name()
                dt = pmml.get_DataDictionary()
                for dd in dt.get_DataField():
                    if dd.get_name() == target_name:
                        val = dd.get_Value()
                        for vv in val:
                            if dd.get_dataType() == 'integer':
                                classes.append(int(vv.get_value()))
                            else:
                                classes.append(vv.get_value())
        return fields, classes


    def predict(self,X):
        predictions = list()
        results = list()
        # print('multiple ',self.multiple_model_method)
        if self.multiple_model_method == 'modelChain':
            for idx, tt in enumerate(self.trees):
                ppp = np.array([0.0 for i in range(len(X))])
                for t in tt:
                    ppp += t.predict(X)
                if len(self.transformedOutputs) != 0:
                    for index in range(len(self.transformedOutputs[idx])-1,-1,-1):
                        func, const = self.transformedOutputs[idx][index]
                        if func == '*':
                            ppp *= float(const)
                        elif func == '+':
                            ppp += float(const)
                predictions.append(ppp)
        else:
            for tt in self.trees:
                predictions.append(tt.predict(X))
        if self.multiple_model_method == 'majorityVote':
            for i in range(len(predictions[0])):
                res = list()
                for j in range(len(predictions)):
                    res.append(predictions[j][i])
                results.append(max(res, key=res.count))
        elif self.multiple_model_method == 'average':
            for i in range(len(predictions[0])):
                res = 0
                for j in range(len(predictions)):
                    res += predictions[j][i]
                res /= len(predictions)
                results.append(res)
        elif self.multiple_model_method == 'sum':
            for i in range(len(predictions[0])):
                res = 0
                for j in range(len(predictions)):
                    res += predictions[j][i]
                if self.rescaleRequired==True:
                    if not self.rescaleFactor:
                        self.rescaleFactor = 1.0
                    if not self.rescaleConstant:
                        self.rescaleConstant = 0.0
                    results.append(res * self.rescaleFactor + self.rescaleConstant)
                else:
                    results.append(res)

        elif self.multiple_model_method == 'modelChain':
            for i in range(len(predictions[0])):
                res = list()
                for j in range(len(predictions)):
                    res.append(predictions[j][i])
                res = np.array(res)
                if self.normalizationMethod == 'logit':
                    res *= -1
                    res = np.exp(res)
                    res += 1
                    res = 1/res
                if len(res) == 1:
                    res_prob = np.array([1 - res[0], res[0]])
                    results.append(self.classes[np.argmax(res_prob)])
                else:
                    res_arr = np.array(res)
                    results.append(self.classes[np.argmax(res_arr)])
        return np.array(results)