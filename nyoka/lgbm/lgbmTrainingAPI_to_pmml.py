from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from pprint import pprint
from PMML43Ext import *
from datetime import datetime
import metadata as md

def ExportToPMML(model,tasktype,target_name):
    jsondata = model.dump_model()
    feature_names = jsondata['feature_names']
    objective = jsondata['objective']
    tree_info = jsondata['tree_info']
    pandas_categorical = jsondata['pandas_categorical']
    num_class  =jsondata['num_class']
    regressionObjectiveList = ['regression', 'regression_l1', 'regression_l2', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie']
    clasificationObjectiveList = ['binary', 'multiclass', 'multiclassova']
    otherObjectiveList = ['xentropy', 'xentlambda', 'lambdarank']
    left_child_op = {"==":"equal", "<=":"lessOrEqual", ">=":"greaterOrEqual"}
    right_child_op = {"==":"notEqual", "<=":"greaterThan", ">=":"lessThan"}
    defaultChild = {"True":"Left", "False":"Right"}
    objectiveSplit = objective.split()[0]
    functionName = 'regression' if objectiveSplit in regressionObjectiveList else 'classification' if objectiveSplit in clasificationObjectiveList else 'mixed'

    def get_MiningModel(tree_information = None, features = None):
        #MiningSc = MiningSchema(MiningField=[MiningField(name = feature, lowValue = data[feature].min(), highValue = data[feature].max() ) for feature in features])
        # MiningSc = MiningSchema(MiningField=[MiningField(name = feature, optype="continuous") for feature in features])
        mf = [MiningField(name = feature, optype="continuous") for feature in features]
        mf.append(MiningField(name = target_name, usageType = "target", optype="continuous"))
        seg = get_Segmentation(tree_information=tree_information, features = features)
        mm = MiningModel(modelName="LightGBModel", algorithmName="LightGBM", functionName = functionName, MiningSchema= MiningSchema(MiningField=mf), Segmentation= seg,taskType=tasktype)
        mm.set_Extension([Extension(name='objective', value=objective),Extension(name='pandas_categorical', value=pandas_categorical),Extension(name='num_class', value=num_class)])
        return mm
    
    def get_Segmentation(tree_information=None, features = None):
        segmentation_list = list()
        for i in range(len(tree_information)):
            tree_data = tree_information[i]
            tree = get_TreeModel(treeData=tree_data, features = features)
            segmentation_list.append(Segment(id=i+1, TreeModel=tree))
        segmentation = Segmentation(Segment=segmentation_list)
        return segmentation

    def get_TreeModel(treeData = None, features = None):
        #ms = MiningSchema(MiningField=[MiningField(name = feature, lowValue = data[feature].min(), highValue = data[feature].max() ) for feature in features])
        ms = MiningSchema(MiningField=[MiningField(name = feature, optype="continuous") for feature in features])
        node = get_Tree(treeData=treeData['tree_structure'],features = features)
        tree = TreeModel(modelName="DecisionTreeModel", functionName=functionName, MiningSchema=ms, Node=node)
        tree.set_Extension([Extension(name='shrinkage', value=treeData['shrinkage'])])
        return tree

    def get_Tree(treeData = None, features = None):
        rootNode = Node()
        create_node(treeData,rootNode,features)
        return rootNode

    def create_node(obj, main_node,derived_col_names):

        def create_left_node(obj,derived_col_names):
            nd = Node()
            nd.set_SimplePredicate(
                SimplePredicate(field=derived_col_names[int(obj['split_feature'])], operator=left_child_op[obj['decision_type']], value=obj['threshold']))
            create_node(obj['left_child'], nd, derived_col_names)
            return nd

        def create_right_node(obj,derived_col_names):
            nd = Node()
            nd.set_SimplePredicate(
                SimplePredicate(field=derived_col_names[int(obj['split_feature'])], operator=right_child_op[obj['decision_type']], value=obj['threshold']))
            create_node(obj['right_child'], nd, derived_col_names)
            return nd

        if 'leaf_index' in obj:
            main_node.set_score(obj['leaf_value'])
            main_node.set_recordCount(obj['leaf_count'])
            main_node.set_id(obj['leaf_index'])
        elif 'split_index' in obj:
            main_node.set_score(obj['internal_value'])
            main_node.set_recordCount(obj['internal_count'])
            main_node.set_id(obj['split_index'])
            main_node.set_defaultChild(defaultChild[str(obj['default_left'])])
            main_node.set_Extension([Extension(name='gain', value=obj['split_gain']), Extension(name='missing_type', value=obj['missing_type'])])
            main_node.add_Node(create_left_node(obj,derived_col_names))
            main_node.add_Node(create_right_node(obj,derived_col_names))

    return {'MiningModel': [get_MiningModel(tree_information=tree_info, features = feature_names)]}

