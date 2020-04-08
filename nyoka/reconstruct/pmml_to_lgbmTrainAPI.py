from nyoka.PMML43Ext import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
th_count = 0 
global_threshold_count = 0

def reconstruct(nyoka_pmml_obj):
    
    def get_tree_string(segmentation=None, temp_file = None):
        segment_list = segmentation.get_Segment()
        for segment in segment_list:
            get_string_tree_data(segment=segment,file=temp_file)
        temp_file.write("end of trees\n\n")

    def get_string_tree_data(segment=None,file=None):
        global th_count
        global global_threshold_count
        tree_model = segment.get_TreeModel()
        root_node = tree_model.get_Node()
        decision_type_mapper = {"equal":{"Left":{},
                                         "Right":{"None": 1 , "NaN": 9}},
                                "lessOrEqual":{"Left":{"None": 2 , "NaN": 10},
                                               "Right":{"NaN": 8}},
                                "greaterOrEqual":{}}
        split_feature=dict()
        split_gain=dict()
        threshold=dict()
        cat_threshold=dict()
        decision_type=dict()
        left_child=list()
        right_child=list()
        leaf_value=dict()
        leaf_count=dict()
        internal_value=dict()
        internal_count=dict()
        internodeList = dict()
        global_threshold_count = 0
        th_count = 0

        def get_gain_missing_type_and_node_info(Node = None):
            gain_ = Node.get_gain()
            missing_type_ = Node.get_missingType()
            intermediate_node_ = True if gain_ else False
            return gain_, missing_type_, intermediate_node_
            # ExtensionList = Node.get_Extension()
            # intermediate_node = False
            # if ExtensionList:
            #     intermediate_node = True
            #     for extension in ExtensionList:
            #         if extension.get_name() == 'gain':
            #             gain = np.float64(extension.get_value())
            #         elif extension.get_name() == 'missing_type':
            #             missing_type = extension.get_value()
            #     return gain, missing_type, intermediate_node
            # return None, None, intermediate_node

        def get_child_node_data(node = None, side = None):
            global global_threshold_count
            if node is not None:
                gain , missing_type, node_info = get_gain_missing_type_and_node_info(Node = node)
                node_id = int(node.get_id())
                if node_info:
                    if gain in internodeList.keys():
                        internodeList[gain].append(node)
                    else:
                        internodeList[gain]=[node]
                    split_gain[node_id] = gain
                    internal_count[node_id] = int(node.get_recordCount())
                    internal_value[node_id] = node.get_score()
                    SimplePredicate = node.get_Node()[0].get_SimplePredicate()
                    if SimplePredicate is not None:
                        split_feature[node_id] = features.index(SimplePredicate.get_field())
                        decision_type[node_id] = decision_type_mapper[SimplePredicate.get_operator()][node.get_defaultChild()][missing_type]
                        temp_threshold = SimplePredicate.get_value()
                        threshold[node_id] = temp_threshold
                        if(temp_threshold == '0' or temp_threshold == '1'):
                            global_threshold_count = global_threshold_count+1
                            cat_threshold[node_id] = int(temp_threshold)+1

                        # if('||' in temp_threshold):                        #Remove this
                        #     global_threshold_count = global_threshold_count+1
                        #     # cat_threshold[node_id] = int(temp_threshold)+1
                        
                    if side=="left":
                        left_child.append(node_id)
                    elif side=="right":
                        right_child.append(node_id)
                else:
                    leaf_count[node_id] = int(node.get_recordCount())
                    leaf_value[node_id] = node.get_score()
                    if side=="left":
                        left_child.append(-1*(node_id+1))
                    elif side=="right":
                        right_child.append(-1*(node_id+1))
                        
        def extractChild(node =None):
            childs = node.get_Node()
            get_child_node_data(childs[0], side = "left")
            get_child_node_data(childs[1], side = "right")
            
        def extractAll(node = None):
            get_child_node_data(node,side=None)
            notEnd = True
            while notEnd:
                try:
                    max_ = max(internodeList.keys())
                    for node in internodeList[max_]:
                        extractChild(node)
                    internodeList.pop(max_)
                except:
                    notEnd = False

        def asignnum(): 
            global th_count
            rc = th_count
            th_count = th_count+1
            return rc

        if(root_node.get_id()):
            extractAll(root_node)
        else:
            leaf_value[0] = 0
        
        # shrinkage = tree_model.get_Extension()[0].get_value() if tree_model.get_Extension()[0].get_name() == 'shrinkage' else None
        shrinkage = str(tree_model.get_shrinkage())
        file.write("Tree="+str(int(segment.get_id())-1)+"\n")
        file.write("num_leaves="+str(len(leaf_value))+"\n")
        file.write("num_cat="+str(global_threshold_count)+"\n")
        file.write("split_feature="+" ".join(map(str, [split_feature[i] for i in sorted (split_feature)]))+"\n")
        file.write("split_gain="+" ".join(map(str, [split_gain[i] for i in sorted (split_gain)]))+"\n")
        file.write("threshold="+" ".join(map(str, [threshold[i] if threshold[i] not in ['0','1'] else asignnum() for i in sorted (threshold)]))+"\n")
        file.write("decision_type="+" ".join(map(str, [decision_type[i] for i in sorted (decision_type)]))+"\n")
        file.write("left_child="+" ".join(map(str, left_child))+"\n")
        file.write("right_child="+" ".join(map(str, right_child))+"\n")
        file.write("leaf_value="+" ".join(map(str, [leaf_value[i] for i in sorted (leaf_value)]))+"\n")
        file.write("leaf_count="+" ".join(map(str, [leaf_count[i] for i in sorted (leaf_count)]))+"\n")
        file.write("internal_value="+" ".join(map(str, [internal_value[i] for i in sorted (internal_value)]))+"\n")
        file.write("internal_count="+" ".join(map(str, [internal_count[i] for i in sorted (internal_count)]))+"\n")
        if(global_threshold_count>0):
            file.write("cat_boundaries="+" ".join(map(str, list(range(0,global_threshold_count+1))))+"\n")
            file.write("cat_threshold="+" ".join(map(str, [cat_threshold[i] for i in sorted (cat_threshold)]))+"\n")
        file.write("shrinkage="+shrinkage+"\n")
        file.write("\n\n")
        
    # nyoka_pmml = parse(pmml_file_name, silence=True)
    mining_model_obj = nyoka_pmml_obj.MiningModel[0]
    num_class_, pandas_categorical_ = '1' , '[]'
    objective_ = mining_model_obj.get_objective()
    num_class_ = mining_model_obj.get_numberOfClass()
    num_class_ = str(num_class_) if num_class_ else "1"
    pandas_categorical_ = mining_model_obj.get_Extension()[0].get_value()   #Change This
    # mining_model_extension_list = mining_model_obj.get_Extension()
    # num_class, pandas_categorical = '1' , '[]'
    # for ext in mining_model_extension_list:
    #     if ext.get_name() == 'objective':
    #         objective = ext.get_value()
    #     elif ext.get_name() == 'num_class':
    #         num_class = ext.get_value()
    #     elif ext.get_name() == 'pandas_categorical':
    #         pandas_categorical = ext.get_value()
    mf = mining_model_obj.get_MiningSchema().get_MiningField()
    features = list()
    feature_infos = list()
    for field in mf:
        if (field.usageType!="target"):
            features.append(field.get_name())
            feature_infos.append("["+str(field.get_lowValue())+":"+str(field.get_highValue())+"]")
    segmentation_obj = mining_model_obj.Segmentation
    filename = "tempfile_iFVMcrUrCQesaRbHubGi.txt"
    f = open(filename, "w+")
    f.write("tree\n"+
            "version=v2\n"+
            "num_class="+num_class_+"\n"+
            "num_tree_per_iteration="+num_class_+"\n"+
            "label_index=0\n"+
            "max_feature_idx="+str(len(features)-1)+"\n"+
            "objective="+objective_+"\n"+
            "feature_names="+" ".join(features)+"\n"+
            "feature_infos="+" ".join(feature_infos)+"\n"+  #feature_infos is minimum value to maximum value ratio of every features
            #tree_sizes=??????????
            "\n")
    get_tree_string(segmentation=segmentation_obj, temp_file = f)
    f.write("pandas_categorical:"+pandas_categorical_+"\n")
    f.close()
    newgbm = lgb.basic.Booster(params = {'model_str' : open(filename, "r").read()})
    f.close()
    os.remove(filename)
    return newgbm

