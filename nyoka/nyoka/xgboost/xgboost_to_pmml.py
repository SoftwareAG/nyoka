from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import PMML43Ext as pml
import json
import nyoka.nyoka.skl.skl_to_pmml as sklToPmml
from skl import pre_process as pp
from datetime import datetime


def xgboost_to_pmml(pipeline, col_names, target_name, pmml_f_name='from_sklearn.pmml'):
    """
    Exports scikit-learn pipeline object into pmml

    Parameters
    ----------
    pipeline :
        Contains an instance of Pipeline with preprocessing and final estimator
    col_names : List
        Contains list of feature/column names.
    target_name : String
        Name of the target column.
    pmml_f_name : String
        Name of the pmml file. (Default='from_sklearn.pmml')

    Returns
    -------
    Returns a pmml file

    """
    try:
        skl_model = pipeline.steps[-1][1]
    except:
        raise TypeError("Exporter expects pipeleine_instance and not an estimator_instance")
    else:
        if isinstance(col_names, np.ndarray):
            col_names = col_names.tolist()
        ppln_sans_predictor = pipeline.steps[:-1]
        trfm_dict_kwargs = dict()
        derived_col_names = col_names
        categoric_values = tuple()
        mining_imp_val = tuple()
        if ppln_sans_predictor:
            pml_pp = pp.get_preprocess_val(ppln_sans_predictor, col_names)
            trfm_dict_kwargs['TransformationDictionary'] = pml_pp['trfm_dict']
            derived_col_names = pml_pp['derived_col_names']
            col_names = pml_pp['preprocessed_col_names']
            categoric_values = pml_pp['categorical_feat_values']
            mining_imp_val = pml_pp['mining_imp_values']
        PMML_kwargs = get_PMML_kwargs(skl_model,
                                      derived_col_names,
                                      col_names,
                                      target_name,
                                      mining_imp_val,
                                      categoric_values)
        pmml = pml.PMML(
            version=sklToPmml.get_version(),
            Header=sklToPmml.get_header(),
            DataDictionary=sklToPmml.get_data_dictionary(skl_model, col_names, target_name, categoric_values),
            **trfm_dict_kwargs,
            **PMML_kwargs
        )
        pmml.export(outfile=open(pmml_f_name, "w"), level=0)

def get_PMML_kwargs(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values):
    """
     It returns all the pmml elements.

    Parameters
    ----------
    skl_model :
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
    algo_kwargs = {'MiningModel': get_ensemble_models(skl_model,
                                                      derived_col_names,
                                                      col_names,
                                                      target_name,
                                                      mining_imp_val,
                                                      categoric_values)}
    return algo_kwargs

def get_ensemble_models(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values):
    """
    It returns the Mining Model element of the model

    Parameters
    ----------
    skl_model :
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
    model_kwargs = sklToPmml.get_model_kwargs(skl_model, col_names, target_name, mining_imp_val)
    if 'XGBRegressor' in str(skl_model.__class__):
        model_kwargs['Targets'] = sklToPmml.get_targets(skl_model, target_name)
    mining_models = list()
    mining_models.append(pml.MiningModel(
        Segmentation=get_outer_segmentation(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values),
        **model_kwargs
    ))
    return mining_models



def get_outer_segmentation(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values):
    """
    It returns the Segmentation element of the model.

    Parameters
    ----------
    skl_model :
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

    if 'XGBRegressor' in str(skl_model.__class__):
        segmentation=get_segments(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values)
    else:
        segmentation = pml.Segmentation(
            multipleModelMethod=get_multiple_model_method(skl_model),
            Segment=get_segments(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values)
        )
    return segmentation

def get_segments(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values):
    """
    It returns the Segment element of the model.

   Parameters
   ----------
   skl_model :
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
    if 'XGBClassifier'  in str(skl_model.__class__):
        segments=get_segments_for_xgbc(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values)
    elif 'XGBRegressor' in str(skl_model.__class__):
        segments=get_segments_for_xgbr(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values)
    return segments

def get_segments_for_xgbr(skl_model, derived_col_names, feature_names, target_name, mining_imp_val,categorical_values):
    """
        It returns all the Segments element of the model

       Parameters
       ----------
       skl_model :
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
    for i in range(skl_model.n_estimators):
        get_nodes_in_json_format.append(json.loads(skl_model._Booster.get_dump(dump_format='json')[i]))
    main_key_value = generate_main_Key_Value(get_nodes_in_json_format)
    segmentation = pml.Segmentation(multipleModelMethod="sum",
                                    Segment=generate_Segments_Equal_To_Estimators(main_key_value, derived_col_names,
                                                                                  feature_names))
    return segmentation

def node_generator(dict_var):
    """
    This method yields all the nodes in a structured format

    Parameters
    ----------
    dict_var: Dictionary
        Contains a dictionary of JSON-format of the nodes.
    Yield:
    -------
        Yields a list of nodes in a structured format.
    """
    for k, v in dict_var.items():
        if k == "split_condition":
            yield str(v)+' split_condition '+str(dict_var.get('split'))
        elif k == "leaf":
            yield str(v)+' score'

        elif isinstance(v, list):
            for i in range(len(v)-1,-1,-1):
                for id_val in node_generator(v[i]):
                    yield id_val

def generate_main_Key_Value(fetch):
    """
    It returns a List where the nodes of the model are in a structured format.

   Parameters
   ----------
   fetch: List
       Contains nodes in JSON format
   Returns:
   -------
   main_key_value:
        Returns a list of nodes in a structured format.
   """
    main_key_value = []
    for i in range(len(fetch)):
        key_value = []
        for k in node_generator(fetch[i]):
            key_value.append(k)
        if len(key_value) > 1:
            main_key_value.append(key_value)
    return main_key_value

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
    main_node_list = []
    node = []
    for i, all_segments in zip(range(len(val)), val):
        main_node = pml.Node(True_=pml.True_())
        mining_field_for_innner_segments = col_names
        m_flds = []
        for each_string in range(len(all_segments) - 1):
            words = all_segments[each_string]
            words = words.split(' ', 2)
            if len(words) >= 3:
                node_ = pml.Node()
                node_.set_SimplePredicate(
                    pml.SimplePredicate(field=replace_name_with_derivedColumnNames(words[2], derived_col_names),
                                        operator="greaterOrEqual", value=words[0]))
                node.append(node_)
            elif len(words) == 2:
                node[-1].set_score(words[0])
                if len(node) == 1:
                    main_node.add_Node(node[0])
                    del node[0]
                else:
                    node[-2].add_Node(node[-1])
                    del node[-1]

        last_string = all_segments[-1].split(' ')
        main_node.set_score(last_string[0])
        main_node_list.append(main_node)

        for name in mining_field_for_innner_segments:
            m_flds.append(pml.MiningField(name=name))

        segments_equal_to_estimators.append((pml.Segment(id=i + 1, True_=pml.True_(),
                                                     TreeModel=pml.TreeModel(functionName="regression",
                                                                         missingValueStrategy="none",
                                                                         noTrueChildStrategy="returnLastPrediction",
                                                                         splitCharacteristic="multiSplit",
                                                                         Node=main_node,
                                                                         MiningSchema=pml.MiningSchema(
                                                                             MiningField=m_flds)))))

    return segments_equal_to_estimators

def add_segmentation(skl_model,segments_equal_to_estimators,mining_schema_for_1st_segment,out,id):
    """
    It returns the First Segments for a binary classifier and returns number of Segments equls to number of values
    target class for multiclass classifier

    Parameters
    ----------
    skl_model:
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
    mining_model = pml.MiningModel(functionName='regression', MiningSchema=mining_schema_for_1st_segment,
                                         Output=out, Segmentation=segmentation)
    if skl_model.n_classes_==2:
        First_segment = pml.Segment(True_=pml.True_(), id=id, MiningModel=mining_model)
        return First_segment
    else:
        segments_equal_to_class = pml.Segment(True_=pml.True_(), id=id + 1, MiningModel=mining_model)
        return segments_equal_to_class




def get_segments_for_xgbc(skl_model, derived_col_names, feature_names, target_name, mining_imp_val,categoric_values):
    """
    It returns all the segments of the Xgboost classifier.

    Parameters
    ----------
    skl_model :
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

    if skl_model.n_classes_ == 2:
        get_nodes_in_json_format=[]
        for i in range(skl_model.n_estimators):
            get_nodes_in_json_format.append(json.loads(skl_model._Booster.get_dump(dump_format='json')[i]))
        main_key_value = generate_main_Key_Value(get_nodes_in_json_format)
        mining_schema_for_1st_segment = mining_Field_For_First_Segment(feature_names)
        outputField = list()
        outputField.append(pml.OutputField(name="xgbValue", optype="continuous", dataType="float",
                                           feature="predictedValue", isFinalResult="true"))
        out = pml.Output(OutputField=outputField)
        oField=list()
        oField.append('xgbValue')
        segments_equal_to_estimators = generate_Segments_Equal_To_Estimators(main_key_value, derived_col_names,
                                                                             feature_names)
        First_segment = add_segmentation(skl_model,segments_equal_to_estimators, mining_schema_for_1st_segment, out, 1)
        last_segment = pml.Segment(True_=pml.True_(), id=2,
                                   RegressionModel=sklToPmml.get_regrs_models(skl_model, oField, oField, target_name,
                                                                    mining_imp_val,categoric_values)[0])
        segments.append(First_segment)

        segments.append(last_segment)
    else:

        get_nodes_in_json_format = []
        for i in range(skl_model.n_estimators * skl_model.n_classes_):
            get_nodes_in_json_format.append(json.loads(skl_model._Booster.get_dump(dump_format='json')[i]))
        main_key_value = generate_main_Key_Value(get_nodes_in_json_format)
        oField = list()
        for index in range(0, skl_model.n_classes_):
            inner_segment = []
            for in_seg in range(index, len(main_key_value), skl_model.n_classes_):
                inner_segment.append(main_key_value[in_seg])
            mining_schema_for_1st_segment = mining_Field_For_First_Segment(feature_names)
            outputField = list()
            outputField.append(pml.OutputField(name='xgbValue(' + str(index) + ')', optype="continuous",
                                      feature="predictedValue", isFinalResult="true"))
            out = pml.Output(OutputField=outputField)

            oField.append('xgbValue(' + str(index) + ')')
            segments_equal_to_estimators = generate_Segments_Equal_To_Estimators(inner_segment, derived_col_names,
                                                                                 feature_names)
            segments_equal_to_class = add_segmentation(skl_model,segments_equal_to_estimators,
                                                       mining_schema_for_1st_segment, out, index)
            segments.append(segments_equal_to_class)
        last_segment = pml.Segment(True_=pml.True_(), id=skl_model.n_classes_ + 1,
                                   RegressionModel=sklToPmml.get_regrs_models(skl_model,oField,oField,target_name,
                                                                    mining_imp_val,categoric_values)[0])
        segments.append(last_segment)
    return segments

def get_multiple_model_method(skl_model):
    """
    It returns the name of the Multiple Model Chain element of the model.

    Parameters
    ----------
    skl_model :
        Contains Xgboost model object
    Returns
    -------
    modelChain for XGBoost Classifier,
    sum for XGboost Regressor,

    """
    if 'XGBClassifier' in str(skl_model.__class__):
        return 'modelChain'
    else:
        return 'sum'

