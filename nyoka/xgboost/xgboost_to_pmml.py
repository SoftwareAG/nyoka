from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import PMML43Ext as pml
import json
#import nyoka.skl.skl_to_pmml as skl_to_pmml
from skl import pre_process as pp
from datetime import datetime
from nyoka.skl import skl_to_pmml


def xgboost_to_pmml(model,derived_col_names,col_names,target_name,mining_imp_val,categoric_values,tasktype,modelPath):
    """
    Exports xgboost pipeline object into pmml

    Parameters
    ----------
    pipeline :
        Contains an instance of Pipeline with preprocessing and final estimator
    col_names : List
        Contains list of feature/column names.
    target_name : String
        Name of the target column.
    pmml_f_name : String
        Name of the pmml file. (Default='from_xgboost.pmml')

    Returns
    -------
    Returns a pmml file

    """
    PMML_kwargs = get_PMML_kwargs(model,
                                    derived_col_names,
                                    col_names,
                                    target_name,
                                    mining_imp_val,
                                    categoric_values,tasktype,modelPath)
    PMML_kwargs['MiningModel'][0].__dict__['taskType']=tasktype
    return PMML_kwargs

def get_PMML_kwargs(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath):
    """
     It returns all the pmml elements.

    Parameters
    ----------
    model :
        Contains XGBoost model object.
    derived_col_names : List
        Contains column names after preprocessing
    col_names : List
        Contains list of feature/column names.
    target_name : String
        Name of the target column .
    mining_imp_val : tuple
        Contains the mining_attributes,mining_strategy, mining_impute_value
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    algo_kwargs : { dictionary element}
        Get the PMML model argument based on XGBoost model object
    """
    algo_kwargs = {'MiningModel': get_ensemble_models(model,
                                                      derived_col_names,
                                                      col_names,
                                                      target_name,
                                                      mining_imp_val,
                                                      categoric_values,tasktype,modelPath)}
    return algo_kwargs

def get_ensemble_models(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath):
    """
    It returns the Mining Model element of the model

    Parameters
    ----------
    model :
        Contains Xgboost model object.
    derived_col_names : List
        Contains column names after preprocessing.
    col_names : List
        Contains list of feature/column names.
    target_name : String
        Name of the Target column.
    mining_imp_val : tuple
        Contains the mining_attributes,mining_strategy, mining_impute_value.
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    mining_models :
        Returns the MiningModel of the respective Xgboost model
    """
    model_kwargs = skl_to_pmml.get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values)
    if 'XGBRegressor' in str(model.__class__):
        model_kwargs['Targets'] = skl_to_pmml.get_targets(model, target_name)
    mining_models = list()
    mining_models.append(pml.MiningModel(
        modelName="XGBoostModel",
        Segmentation=get_outer_segmentation(model, col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath),
        **model_kwargs
    ))
    return mining_models



def get_outer_segmentation(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath):
    """
    It returns the Segmentation element of the model.

    Parameters
    ----------
    model :
        Contains Xgboost model object.
    derived_col_names : List
        Contains column names after preprocessing.
    col_names : List
        Contains list of feature/column names.
    target_name : String
        Name of the Target column.
    mining_imp_val : tuple
        Contains the mining_attributes,mining_strategy, mining_impute_value
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    segmentation :
        Get the outer most Segmentation of an xgboost model

    """

    if 'XGBRegressor' in str(model.__class__):
        segmentation=get_segments(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath)
    else:
        segmentation = pml.Segmentation(
            multipleModelMethod=get_multiple_model_method(model),
            Segment=get_segments(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath)
        )
    return segmentation

def get_segments(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath):
    """
    It returns the Segment element of the model.

   Parameters
   ----------
   model :
       Contains Xgboost model object.
   derived_col_names : List
       Contains column names after preprocessing.
   col_names : List
       Contains list of feature/column names.
   target_name : String
       Name of the Target column.
   mining_imp_val : tuple
        Contains the mining_attributes,mining_strategy, mining_impute_value
    categoric_values : tuple
        Contains Categorical attribute names and its values

   Returns
   -------
   segment :
       Get the Segments for the Segmentation element.

   """
    segments = None
    if 'XGBClassifier'  in str(model.__class__):
        segments=get_segments_for_xgbc(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath)
    elif 'XGBRegressor' in str(model.__class__):
        segments=get_segments_for_xgbr(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath)
    return segments

def get_segments_for_xgbr(model, derived_col_names, feature_names, target_name, mining_imp_val,categorical_values,tasktype,modelPath):
    """
        It returns all the Segments element of the model

       Parameters
       ----------
       model :
           Contains Xgboost model object.
       derived_col_names : List
           Contains column names after preprocessing.
       feature_names : List
           Contains list of feature/column names.
       target_name : List
           Name of the Target column.
       mining_imp_val : tuple
            Contains the mining_attributes,mining_strategy, mining_impute_value
        categoric_values : tuple
            Contains Categorical attribute names and its values

       Returns
       -------
       segment :
           Get the Segmentation element which contains inner segments.

       """
    segments = list()
    get_nodes_in_json_format = []
    for i in range(model.n_estimators):
        get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))
    segmentation = pml.Segmentation(multipleModelMethod="sum",
                                    Segment=generate_Segments_Equal_To_Estimators(get_nodes_in_json_format, derived_col_names,
                                                                                  feature_names))
    return segmentation


def mining_Field_For_First_Segment(feature_names):
    """
        It returns the Mining Schema of the First Segment.

      Parameters
      ----------
      feature_names: List
          Contains list of feature/column names.
      Returns:
      -------
      mining_schema_for_1st_segment:
           Returns the MiningSchema for the main segment.
      """
    mining_fields_1st_segment = []
    for name in feature_names:
        mining_fields_1st_segment.append(pml.MiningField(name=name))
    mining_schema_for_1st_segment = pml.MiningSchema(MiningField=mining_fields_1st_segment)
    return mining_schema_for_1st_segment

def replace_name_with_derivedColumnNames(original_name, derived_col_names):
    """
    It replace the default names with the names of the attributes.

     Parameters
     ----------
     original_name: List
         The name of the node retrieve from model
     derived_col_names: List
        The name of the derived attributes.
     Returns:
     -------
     col_name:
          Returns the derived column name/original column name.
     """
    new = str.replace(original_name, 'f', '')
    if new.isdigit():
        col_name = derived_col_names[int(new)]
    else:
        col_name = original_name
    return col_name


def create_node(obj, main_node,derived_col_names):
    """
    It creates nodes.

    Parameters
    ----------
    obj: Json
        Contains nodes in json format.
    main_node:
        Contains node build with Nyoka class.
    derived_col_names: List
        Contains column names after preprocessing.
    """
    def create_left_node(obj,derived_col_names):
        nd = pml.Node()
        nd.set_SimplePredicate(
            pml.SimplePredicate(field=replace_name_with_derivedColumnNames(obj['split'], derived_col_names), operator='lessThan', value=obj['split_condition']))
        create_node(obj['children'][0], nd, derived_col_names)
        return nd

    def create_right_node(obj,derived_col_names):
        nd = pml.Node()
        nd.set_SimplePredicate(
            pml.SimplePredicate(field=replace_name_with_derivedColumnNames(obj['split'], derived_col_names), operator='greaterOrEqual', value=obj['split_condition']))
        create_node(obj['children'][1], nd, derived_col_names)
        return nd

    if 'split' not in obj:
        main_node.set_score(obj['leaf'])
    else:

        main_node.add_Node(create_left_node(obj,derived_col_names))
        main_node.add_Node(create_right_node(obj,derived_col_names))


def generate_Segments_Equal_To_Estimators(val, derived_col_names, col_names):
    """
    It returns number of Segments equal to the estimator of the model.

    Parameters
    ----------
    val: List
        Contains a list of well structured node for binary classification/inner segments for multi-class classification
    derived_col_names: List
        Contains column names after preprocessing.
    col_names: List
        Contains list of feature/column names.
    Returns:
    -------
    segments_equal_to_estimators:
         Returns list of segments equal to number of estimator of the model
    """
    segments_equal_to_estimators = []
    for i in range(len(val)):
        main_node = pml.Node(True_=pml.True_())
        m_flds = []
        mining_field_for_innner_segments = col_names
        create_node(val[i], main_node, derived_col_names)

        for name in mining_field_for_innner_segments:
            m_flds.append(pml.MiningField(name=name))

        segments_equal_to_estimators.append((pml.Segment(id=i + 1, True_=pml.True_(),
                                                         TreeModel=pml.TreeModel(functionName="regression",
                                                         modelName="DecisionTreeModel",
                                                                                 missingValueStrategy="none",
                                                                                 noTrueChildStrategy="returnLastPrediction",
                                                                                 splitCharacteristic="multiSplit",
                                                                                 Node=main_node,
                                                                                 MiningSchema=pml.MiningSchema(
                                                                                     MiningField=m_flds)))))

    return segments_equal_to_estimators

def add_segmentation(model,segments_equal_to_estimators,mining_schema_for_1st_segment,out,id):
    """
    It returns the First Segments for a binary classifier and returns number of Segments equls to number of values
    target class for multiclass classifier

    Parameters
    ----------
    model:
       Contains Xgboost model object.
    segments_equal_to_estimators: List
        Contains List Segements equals to the number of the estimators of the model.
    mining_schema_for_1st_segment:
        Contains Mining Schema for the First Segment
    out:
        Contains the Output element
    id: Integer
        Index of the Segements

    Returns:
    -------
    segments_equal_to_estimators:
         Returns list of segments equal to number of estimator of the model
    """

    segmentation = pml.Segmentation(multipleModelMethod="sum", Segment=segments_equal_to_estimators)
    mining_model = pml.MiningModel(functionName='regression', modelName="MiningModel", MiningSchema=mining_schema_for_1st_segment,
                                         Output=out, Segmentation=segmentation)
    if model.n_classes_==2:
        First_segment = pml.Segment(True_=pml.True_(), id=id, MiningModel=mining_model)
        return First_segment
    else:
        segments_equal_to_class = pml.Segment(True_=pml.True_(), id=id + 1, MiningModel=mining_model)
        return segments_equal_to_class




def get_segments_for_xgbc(model, derived_col_names, feature_names, target_name, mining_imp_val,categoric_values,tasktype,modelPath):
    """
    It returns all the segments of the Xgboost classifier.

    Parameters
    ----------
    model :
        Contains Xgboost model object.
    derived_col_names : List
        Contains column names after preprocessing.
    feature_names: List
        Contains list of feature/column names.
    target_name : String
        Name of the Target column.
    mining_imp_val : tuple
        Contains the mining_attributes,mining_strategy, mining_impute_value
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    regrs_models :
        Returns all the segments of the xgboost model.
        """
    segments = list()

    if model.n_classes_ == 2:
        get_nodes_in_json_format=[]
        for i in range(model.n_estimators):
            get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))
        mining_schema_for_1st_segment = mining_Field_For_First_Segment(feature_names)
        outputField = list()
        outputField.append(pml.OutputField(name="xgbValue", optype="continuous", dataType="float",
                                           feature="predictedValue", isFinalResult="true"))
        out = pml.Output(OutputField=outputField)
        oField=list()
        oField.append('xgbValue')
        segments_equal_to_estimators = generate_Segments_Equal_To_Estimators(get_nodes_in_json_format, derived_col_names,
                                                                             feature_names)
        First_segment = add_segmentation(model,segments_equal_to_estimators, mining_schema_for_1st_segment, out, 1)
        reg_model=skl_to_pmml.get_regrs_models(model, oField, oField, target_name,mining_imp_val,categoric_values,tasktype,modelPath)[0]
        reg_model.normalizationMethod='logit'
        last_segment = pml.Segment(True_=pml.True_(), id=2,
                                   RegressionModel=reg_model)
        segments.append(First_segment)

        segments.append(last_segment)
    else:

        get_nodes_in_json_format = []
        for i in range(model.n_estimators * model.n_classes_):
            get_nodes_in_json_format.append(json.loads(model._Booster.get_dump(dump_format='json')[i]))
        oField = list()
        for index in range(0, model.n_classes_):
            inner_segment = []
            for in_seg in range(index, len(get_nodes_in_json_format), model.n_classes_):
                inner_segment.append(get_nodes_in_json_format[in_seg])
            mining_schema_for_1st_segment = mining_Field_For_First_Segment(feature_names)
            outputField = list()
            outputField.append(pml.OutputField(name='xgbValue(' + str(index) + ')', optype="continuous",
                                      feature="predictedValue", dataType="float", isFinalResult="true"))
            out = pml.Output(OutputField=outputField)

            oField.append('xgbValue(' + str(index) + ')')
            segments_equal_to_estimators = generate_Segments_Equal_To_Estimators(inner_segment, derived_col_names,
                                                                                 feature_names)
            segments_equal_to_class = add_segmentation(model,segments_equal_to_estimators,
                                                       mining_schema_for_1st_segment, out, index)
            segments.append(segments_equal_to_class)
        reg_model=skl_to_pmml.get_regrs_models(model,oField,oField,target_name,mining_imp_val,categoric_values,tasktype,modelPath)[0]
        reg_model.normalizationMethod='softmax'
        last_segment = pml.Segment(True_=pml.True_(), id=model.n_classes_ + 1,
                                   RegressionModel=reg_model)
        segments.append(last_segment)
    return segments

def get_multiple_model_method(model):
    """
    It returns the name of the Multiple Model Chain element of the model.

    Parameters
    ----------
    model :
        Contains Xgboost model object
    Returns
    -------
    modelChain for XGBoost Classifier,
    sum for XGboost Regressor,

    """
    if 'XGBClassifier' in str(model.__class__):
        return 'modelChain'
    else:
        return 'sum'

