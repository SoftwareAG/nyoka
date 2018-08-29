from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import PMML43Ext as pml
import nyoka.nyoka.skl.skl_to_pmml as sklToPmml
import nyoka.nyoka.xgboost.xgboost_to_pmml as xgboostToPmml
import json
from skl import pre_process as pp
from datetime import datetime



def lgb_to_pmml(pipeline, col_names, target_name, pmml_f_name='from_sklearn.pmml'):
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
        Contains LGB model object.
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
        Get the PMML model argument based on LGB model object
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
        Contains LGB model object.
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
        Returns the MiningModel of the respective LGB model
    """
    model_kwargs = sklToPmml.get_model_kwargs(skl_model, col_names, target_name, mining_imp_val)
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
        Contains LGB model object.
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
        Get the outer most Segmentation of an LGB model

    """

    if 'LGBMRegressor' in str(skl_model.__class__):
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
       Contains LGB model object.
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
    if 'LGBMClassifier' in str(skl_model.__class__):
        segments=get_segments_for_lgbc(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values)
    elif 'LGBMRegressor' in str(skl_model.__class__):
        segments=get_segments_for_lgbr(skl_model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values)
    return segments

def get_segments_for_lgbr(skl_model, derived_col_names, feature_names, target_name, mining_imp_val,categorical_values):
    """
        It returns all the Segments element of the model

       Parameters
       ----------
       skl_model :
           Contains LGB model object.
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
    main_key_value = []
    lgb_dump = skl_model.booster_.dump_model()
    for i in range(len(lgb_dump['tree_info'])):
        tree = lgb_dump['tree_info'][i]['tree_structure']
        list_of_nodes = []
        main_key_value.append(generate_structure_for_lgb(tree, list_of_nodes, derived_col_names))
    segmentation = pml.Segmentation(multipleModelMethod="sum",
                                    Segment=xgboostToPmml.generate_Segments_Equal_To_Estimators(main_key_value, derived_col_names,
                                                                                  feature_names))
    return segmentation




def generate_structure_for_lgb(fetch,main_key_value,derived_col_names):
    """
    It returns a List where the nodes of the model are in a structured format.

    Parameters
    ----------
    fetch : dictionary
        Contains the nodes in dictionary format.

    main_key_value: List
        Empty list used to append the nodes.

    derived_col_names: List
        Contains column names after preprocessing.


    Returns
    -------
    main_key_value :
        Returns the nodes in a structured format inside a list.
    """
    list_of_child=[]
    for k,v in fetch.items():
        if k=='threshold':
            main_key_value.append(str(v)+' split_condition '+str(derived_col_names[int(fetch.get('split_feature'))]))
        if k=='leaf_value':
            main_key_value.append(str(v)+' score')
        if isinstance(v,dict):
            list_of_child.append(v)
    for ii in range(len(list_of_child)-1,-1,-1):
        generate_structure_for_lgb(list_of_child[ii],main_key_value,derived_col_names)
    return main_key_value


def get_segments_for_lgbc(skl_model, derived_col_names, feature_names, target_name, mining_imp_val,categoric_values):
    """
    It returns all the segments of the LGB classifier.

    Parameters
    ----------
    skl_model :
        Contains LGB model object.
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
        Returns all the segments of the LGB model.
        """
    segments = list()

    if skl_model.n_classes_ == 2:
        main_key_value = []
        lgb_dump = skl_model.booster_.dump_model()
        for i in range(len(lgb_dump['tree_info'])):
            tree = lgb_dump['tree_info'][i]['tree_structure']
            list_of_nodes = []
            main_key_value.append(generate_structure_for_lgb(tree, list_of_nodes,derived_col_names))
        mining_schema_for_1st_segment = xgboostToPmml.mining_Field_For_First_Segment(feature_names)
        outputField = list()
        outputField.append(pml.OutputField(name="lgbValue", optype="continuous", dataType="float",
                                           feature="predictedValue", isFinalResult="true"))
        out = pml.Output(OutputField=outputField)
        oField=list()
        oField.append('lgbValue')
        segments_equal_to_estimators = xgboostToPmml.generate_Segments_Equal_To_Estimators(main_key_value, derived_col_names,
                                                                             feature_names)
        First_segment = xgboostToPmml.add_segmentation(skl_model,segments_equal_to_estimators, mining_schema_for_1st_segment, out, 1)
        last_segment = pml.Segment(True_=pml.True_(), id=2,
                                   RegressionModel=sklToPmml.get_regrs_models(skl_model, oField, oField, target_name,
                                                                    mining_imp_val,categoric_values)[0])
        segments.append(First_segment)

        segments.append(last_segment)
    else:
        main_key_value = []
        lgb_dump = skl_model.booster_.dump_model()
        for i in range(len(lgb_dump['tree_info'])):
            tree = lgb_dump['tree_info'][i]['tree_structure']
            list_of_nodes = []
            main_key_value.append(generate_structure_for_lgb(tree, list_of_nodes, derived_col_names))
        oField = list()
        for index in range(0, skl_model.n_classes_):
            inner_segment = []
            for in_seg in range(index, len(main_key_value), skl_model.n_classes_):
                inner_segment.append(main_key_value[in_seg])
            mining_schema_for_1st_segment = xgboostToPmml.mining_Field_For_First_Segment(feature_names)
            outputField = list()
            outputField.append(pml.OutputField(name='lgbValue(' + str(index) + ')', optype="continuous",
                                      feature="predictedValue", isFinalResult="true"))
            out = pml.Output(OutputField=outputField)

            oField.append('lgbValue(' + str(index) + ')')
            segments_equal_to_estimators = xgboostToPmml.generate_Segments_Equal_To_Estimators(inner_segment, derived_col_names,
                                                                                 feature_names)
            segments_equal_to_class = xgboostToPmml.add_segmentation(skl_model,segments_equal_to_estimators,
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
        Contains LGB model object
    Returns
    -------
    modelChain for LGB Classifier,
    sum for LGB Regressor,

    """
    if 'LGBMClassifier' in str(skl_model.__class__):
        return 'modelChain'
    else:
        return 'sum'

