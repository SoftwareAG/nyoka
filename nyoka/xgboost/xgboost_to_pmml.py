"""
 Copyright (c) 2004-2016 Zementis, Inc.
 Copyright (c) 2016-2021 Software AG, Darmstadt, Germany and/or Software AG USA Inc., Reston, VA, USA, and/or its

 SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import PMML44 as pml
import json
import nyoka.skl.skl_to_pmml as sklToPmml
from skl import pre_process as pp
from datetime import datetime
from base.constants import *


def xgboost_to_pmml(pipeline, col_names, target_name, pmml_f_name='from_xgboost.pmml',model_name=None,description=None):
    """
    Exports xgboost model object into pmml

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
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description for the model

    Returns
    -------
    Generates the PMML object and exports it to `pmml_f_name`

    """
    try:
        model = pipeline.steps[-1][1]
    except:
        raise TypeError("Exporter expects pipeleine_instance and not an estimator_instance")
    else:
        if col_names.__class__.__name__ == "ndarray":
            col_names = col_names.tolist()
        ppln_sans_predictor = pipeline.steps[:-1]
        trfm_dict_kwargs = dict()
        derived_col_names = col_names
        categoric_values = tuple()
        mining_imp_val = tuple()
        if ppln_sans_predictor:
            pml_pp = pp.get_preprocess_val(ppln_sans_predictor, col_names, model)
            trfm_dict_kwargs['TransformationDictionary'] = pml_pp['trfm_dict']
            derived_col_names = pml_pp['derived_col_names']
            col_names = pml_pp['preprocessed_col_names']
            categoric_values = pml_pp['categorical_feat_values']
            mining_imp_val = pml_pp['mining_imp_values']
        PMML_kwargs = get_PMML_kwargs(model,
                                      derived_col_names,
                                      col_names,
                                      target_name,
                                      mining_imp_val,
                                      categoric_values,
                                      model_name)
        pmml = pml.PMML(
            version=PMML_SCHEMA.VERSION,
            Header=sklToPmml.get_header(description),
            DataDictionary=sklToPmml.get_data_dictionary(model, col_names, target_name, categoric_values),
            **trfm_dict_kwargs,
            **PMML_kwargs
        )
        pmml.export(outfile=open(pmml_f_name, "w"), level=0)

def get_PMML_kwargs(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):
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
    model_name : string
        Name of the model

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
                                                      categoric_values,
                                                      model_name)}
    return algo_kwargs

def get_ensemble_models(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):
    """
    It returns the Mining Model element of the model

    Parameters
    ----------
    model :
        Contains XGBoost model object.
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
    model_name : string
        Name of the model

    Returns
    -------
    mining_models :
        Returns Nyoka's MiningModel object
    """
    model_kwargs =get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values)
    if 'XGBRegressor' in str(model.__class__):
        model_kwargs['Targets'] =get_targets(model, target_name)
    mining_models = list()
    mining_models.append(pml.MiningModel(
        modelName=model_name if model_name else "XGBoostModel",
        Segmentation=get_outer_segmentation(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name),
        **model_kwargs
    ))
    return mining_models


def get_targets(model, target_name):

    """
    It returns the Target element of the model.

    Parameters
    ----------
    model :
        An Xgboost model instance.
    target_name : String
        Name of the Target column.

    Returns
    -------
    targets :
        Returns Nyoka's Target object
    """
    if model.__class__.__name__ == 'XGBRegressor':
        targets = pml.Targets(
            Target=[
                pml.Target(
                    field=target_name,
                    rescaleConstant="{:.16f}".format(model.base_score if model.base_score is not None
                                                     else 0.5)
                )
            ]
        )
    return targets



def get_outer_segmentation(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):
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
    model_name : string
        Name of the model

    Returns
    -------
    segmentation :
        Returns Nyoka's Segmentation object

    """

    if 'XGBRegressor' in str(model.__class__):
        segmentation=get_segments(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name)
    else:
        segmentation = pml.Segmentation(
            multipleModelMethod=get_multiple_model_method(model),
            Segment=get_segments(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name)
        )
    return segmentation

def get_segments(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):
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
    model_name : string
        Name of the model

    Returns
    -------
    segment :
        Nyoka's Segment object

   """
    segments = None
    if 'XGBClassifier'  in str(model.__class__):
        segments=get_segments_for_xgbc(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name)
    elif 'XGBRegressor' in str(model.__class__):
        segments=get_segments_for_xgbr(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values)
    return segments


def get_segments_for_xgbr(model, derived_col_names, feature_names, target_name, mining_imp_val,categorical_values):
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
        Nyoka's Segment object

    """
    get_nodes_in_json_format = model._Booster.get_dump(dump_format='json')
    segmentation = pml.Segmentation(multipleModelMethod=MULTIPLE_MODEL_METHOD.SUM,
                                    Segment=generate_Segments_Equal_To_Estimators(get_nodes_in_json_format, derived_col_names,
                                                                                  feature_names))
    return segmentation


def mining_Field_For_First_Segment(feature_names):
    """
    It returns the Mining Schema of the First Segment.

    Parameters
    ----------
    feature_names : List
        Contains list of feature/column names.

    Returns
    -------
    mining_schema_for_1st_segment :
        Nyoka's MiningSchema object
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
    original_name : List
        The name of the node retrieve from model
    derived_col_names : List
    The name of the derived attributes.

    Returns
    -------
    col_name :
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
    obj : Json
        Contains nodes in json format.
    main_node :
        Contains node build with Nyoka class.
    derived_col_names : List
        Contains column names after preprocessing.
    """
    def create_left_node(obj,derived_col_names):
        nd = pml.Node()
        nd.set_SimplePredicate(
            pml.SimplePredicate(field=replace_name_with_derivedColumnNames(obj['split'], derived_col_names),\
                 operator=SIMPLE_PREDICATE_OPERATOR.LESS_THAN, value="{:.16f}".format(obj['split_condition'])))
        create_node(obj['children'][0], nd, derived_col_names)
        return nd

    def create_right_node(obj,derived_col_names):
        nd = pml.Node()
        nd.set_SimplePredicate(
            pml.SimplePredicate(field=replace_name_with_derivedColumnNames(obj['split'], derived_col_names),\
                 operator=SIMPLE_PREDICATE_OPERATOR.GREATER_OR_EQUAL, value="{:.16f}".format(obj['split_condition'])))
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
    val : List
        Contains a list of well structured node for binary classification/inner segments for multi-class classification
    derived_col_names : List
        Contains column names after preprocessing.
    col_names : List
        Contains list of feature/column names.

    Returns
    -------
    segments_equal_to_estimators:
        Nyoka's Segment object
    """
    segments_equal_to_estimators = []
    for i in range(len(val)):
        main_node = pml.Node(True_=pml.True_())
        m_flds = []
        mining_field_for_innner_segments = col_names
        create_node(json.loads(val[i]), main_node, derived_col_names)

        for name in mining_field_for_innner_segments:
            m_flds.append(pml.MiningField(name=name))

        segments_equal_to_estimators.append((pml.Segment(id=i + 1, True_=pml.True_(),
                                                         TreeModel=pml.TreeModel(functionName=MINING_FUNCTION.REGRESSION,
                                                         modelName="DecisionTreeModel",
                                                                                 missingValueStrategy="none",
                                                                                 noTrueChildStrategy="returnLastPrediction",
                                                                                 splitCharacteristic=TREE_SPLIT_CHARACTERISTIC.MULTI,
                                                                                 Node=main_node,
                                                                                 MiningSchema=pml.MiningSchema(
                                                                                     MiningField=m_flds)))))

    return segments_equal_to_estimators

def add_segmentation(model,segments_equal_to_estimators,mining_schema_for_1st_segment,out,id):
    """
    It returns segmentation for a mining model

    Parameters
    ----------
    model :
       Contains Xgboost model object.
    segments_equal_to_estimators : List
        Contains List Segements equals to the number of the estimators of the model.
    mining_schema_for_1st_segment :
        Contains Mining Schema for the First Segment
    out :
        Contains the Output element
    id : Integer
        Index of the Segements

    Returns
    -------
    segments_equal_to_estimators:
         Returns Nyoka's Segment object
    """

    segmentation = pml.Segmentation(multipleModelMethod=MULTIPLE_MODEL_METHOD.SUM, Segment=segments_equal_to_estimators)
    mining_model = pml.MiningModel(functionName=MINING_FUNCTION.REGRESSION, modelName="MiningModel", MiningSchema=mining_schema_for_1st_segment,
                                         Output=out, Segmentation=segmentation)
    if model.n_classes_==2:
        First_segment = pml.Segment(True_=pml.True_(), id=id, MiningModel=mining_model)
        return First_segment
    else:
        segments_equal_to_class = pml.Segment(True_=pml.True_(), id=id + 1, MiningModel=mining_model)
        return segments_equal_to_class




def get_segments_for_xgbc(model, derived_col_names, feature_names, target_name, mining_imp_val,categoric_values,model_name):
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
    model_name : string
        Name of the model

    Returns
    -------
    regrs_models :
        Returns Nyoka's Segment object
    """
    segments = list()

    if model.n_classes_ == 2:
        get_nodes_in_json_format=model._Booster.get_dump(dump_format='json')
        mining_schema_for_1st_segment = mining_Field_For_First_Segment(feature_names)
        outputField = list()
        outputField.append(pml.OutputField(name="xgbValue", optype=OPTYPE.CONTINUOUS, dataType=DATATYPE.FLOAT,
                                           feature=RESULT_FEATURE.PREDICTED_VALUE, isFinalResult="true"))
        out = pml.Output(OutputField=outputField)
        oField=list()
        oField.append('xgbValue')
        segments_equal_to_estimators = generate_Segments_Equal_To_Estimators(get_nodes_in_json_format, derived_col_names,
                                                                             feature_names)
        First_segment = add_segmentation(model,segments_equal_to_estimators, mining_schema_for_1st_segment, out, 1)
        reg_model=get_regrs_models(model, oField, oField, target_name,mining_imp_val,categoric_values,model_name)[0]
        reg_model.normalizationMethod=REGRESSION_NORMALIZATION_METHOD.LOGISTIC
        last_segment = pml.Segment(True_=pml.True_(), id=2,
                                   RegressionModel=reg_model)
        segments.append(First_segment)

        segments.append(last_segment)
    else:
        get_nodes_in_json_format = model._Booster.get_dump(dump_format='json')
        oField = list()
        for index in range(0, model.n_classes_):
            inner_segment = []
            for in_seg in range(index, len(get_nodes_in_json_format), model.n_classes_):
                inner_segment.append(get_nodes_in_json_format[in_seg])
            mining_schema_for_1st_segment = mining_Field_For_First_Segment(feature_names)
            outputField = list()
            outputField.append(pml.OutputField(name='xgbValue(' + str(index) + ')', optype=OPTYPE.CONTINUOUS,
                                      feature=RESULT_FEATURE.PREDICTED_VALUE, dataType=DATATYPE.FLOAT, isFinalResult="true"))
            out = pml.Output(OutputField=outputField)

            oField.append('xgbValue(' + str(index) + ')')
            segments_equal_to_estimators = generate_Segments_Equal_To_Estimators(inner_segment, derived_col_names,
                                                                                 feature_names)
            segments_equal_to_class = add_segmentation(model,segments_equal_to_estimators,
                                                       mining_schema_for_1st_segment, out, index)
            segments.append(segments_equal_to_class)
        reg_model=get_regrs_models(model,oField,oField,target_name,mining_imp_val,categoric_values,model_name)[0]
        reg_model.normalizationMethod=REGRESSION_NORMALIZATION_METHOD.SOFTMAX
        last_segment = pml.Segment(True_=pml.True_(), id=model.n_classes_ + 1,
                                   RegressionModel=reg_model)
        segments.append(last_segment)
    return segments


def get_regrs_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,model_name):

    """
    Generates PMML elements for linear models

    Parameters
    ----------
    model :
        An Xgboost model instance.
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
    model_name : string
        Name of the model

    Returns
    -------
    regrs_models : List
        Nyoka's RegressionModel object
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values)
    if model.__class__.__name__ not in ['LinearRegression','LinearSVR']:
        model_kwargs['normalizationMethod'] = REGRESSION_NORMALIZATION_METHOD.LOGISTIC
    regrs_models = list()
    regrs_models.append(pml.RegressionModel(
        modelName=model_name if model_name else model.__class__.__name__,
        RegressionTable=get_regrs_tabl(model, derived_col_names, target_name, categoric_values),
        **model_kwargs
    ))
    return regrs_models


def get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values):

    """
    It returns all the model element for a specific model.

    Parameters
    ----------
    model :
        An instance of an Xgboost model.
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
    model_kwargs : Dictionary
        Returns  function name, MiningSchema and Output of the sk_model object
    """
    model_kwargs = dict()
    model_kwargs['functionName'] = get_mining_func(model)
    model_kwargs['MiningSchema'] = get_mining_schema(model, col_names, target_name, mining_imp_val, categoric_values)
    model_kwargs['Output'] = get_output(model, target_name)

    return model_kwargs



def get_mining_func(model):
    """
    It returns the name of the mining function of the model.

    Parameters
    ----------
    model :
        An Xgboost model instance.

    Returns
    -------
    func_name : String
        Returns the function name of the model

    """
    if hasattr(model, 'n_classes_'):
        if model.n_classes_ > 1:
            func_name = MINING_FUNCTION.CLASSIFICATION
        else:
            func_name = MINING_FUNCTION.REGRESSION
    elif hasattr(model, 'classes_'):
        if len(model.classes_) > 1:
            func_name = MINING_FUNCTION.CLASSIFICATION
        else:
            func_name = MINING_FUNCTION.REGRESSION
    else:
        if hasattr(model, 'n_clusters'):
            func_name = MINING_FUNCTION.CLUSTERING
        else:
            func_name = MINING_FUNCTION.REGRESSION

    return func_name


def get_mining_schema(model, feature_names, target_name, mining_imp_val, categoric_values):

    """
    It returns the Mining Schema of the model.

    Parameters
    ----------
    model :
        An Xgboost model instance.
    feature_names : List
        Contains the list of feature/column name.
    target_name : String
        Name of the Target column.
    mining_imp_val : tuple
        Contains the mining_attributes,mining_strategy, mining_impute_value.
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    MiningSchema :
        Nyoka's MiningSchema object

    """
    if mining_imp_val:
        mining_attributes = mining_imp_val[0]
        mining_strategy = mining_imp_val[1]
        mining_replacement_val = mining_imp_val[2]
    n_features = len(feature_names)
    features_pmml_optype = [OPTYPE.CONTINUOUS] * n_features
    features_pmml_utype = [FIELD_USAGE_TYPE.ACTIVE] * n_features
    target_pmml_utype = FIELD_USAGE_TYPE.TARGET
    mining_func = get_mining_func(model)
    if mining_func == MINING_FUNCTION.CLASSIFICATION:
        target_pmml_optype = OPTYPE.CATEGORICAL
    elif mining_func == MINING_FUNCTION.REGRESSION:
        target_pmml_optype = OPTYPE.CONTINUOUS
    mining_flds = list()
    mining_name_stored = list()
    # handling impute pre processing
    if mining_imp_val:
        for mining_item, mining_idx in zip(mining_attributes, range(len(mining_attributes))):
            for feat_name,feat_idx in zip(feature_names, range(len(feature_names))):
                if feat_name in mining_item:
                    if feat_name not in mining_name_stored:
                        impute_index = mining_item.index(feat_name)

                        mining_flds.append(pml.MiningField(name=str(feat_name),
                                                           optype=features_pmml_optype[feat_idx],
                                                           missingValueReplacement=mining_replacement_val[mining_idx][
                                                              impute_index],
                                                           missingValueTreatment=mining_strategy[mining_idx],
                                                           usageType=features_pmml_utype[feat_idx]))
                        mining_name_stored.append(feat_name)
    if len(categoric_values) > 0:
        for cls_attr in categoric_values[1]:
            mining_flds.append(pml.MiningField(
                name=cls_attr,
                usageType=FIELD_USAGE_TYPE.ACTIVE,
                optype=OPTYPE.CATEGORICAL
            ))
            mining_name_stored.append(cls_attr)
    for feat_name, feat_idx in zip(feature_names, range(len(feature_names))):
        if feat_name not in mining_name_stored:
            mining_flds.append(pml.MiningField(name=str(feat_name),
                                               optype=features_pmml_optype[feat_idx],
                                               usageType=features_pmml_utype[feat_idx]))
    if model.__class__.__name__ not in ['KMeans', 'IsolationForest', 'OneClassSVM']:
        mining_flds.append(pml.MiningField(name=target_name,
                                        optype=target_pmml_optype,
                                            usageType=target_pmml_utype))
    return pml.MiningSchema(MiningField=mining_flds)


def get_output(model, target_name):

    """
    It returns the output element of the model.

    Parameters
    ----------
    model :
        An Xboost model instance.
    target_name : String
        Name of the Target column.

    Returns
    -------
    Output :
        Nyoka's Output object

    """
    mining_func = get_mining_func(model)
    output_fields = list()
    if not has_target(model):
        output_fields.append(pml.OutputField(
                name='predicted',
                feature=RESULT_FEATURE.PREDICTED_VALUE,
                optype=OPTYPE.CONTINUOUS,
                dataType=DATATYPE.DOUBLE
            ))
    else:
        alt_target_name = 'predicted_' + target_name
        if mining_func == MINING_FUNCTION.CLASSIFICATION:
            for cls in model.classes_:
                output_fields.append(pml.OutputField(
                    name='probability_' + str(cls),
                    feature=RESULT_FEATURE.PROBABILITY,
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE,
                    value=str(cls)
                ))
            output_fields.append(pml.OutputField(
                name=alt_target_name,
                feature=RESULT_FEATURE.PREDICTED_VALUE,
                optype=OPTYPE.CATEGORICAL,
                dataType=get_dtype(model.classes_[0])))
        else:
            output_fields.append(pml.OutputField(
                name=alt_target_name,
                feature=RESULT_FEATURE.PREDICTED_VALUE,
                optype=OPTYPE.CONTINUOUS,
                dataType=DATATYPE.DOUBLE))
    return pml.Output(OutputField=output_fields)


def get_regrs_tabl(model, feature_names, target_name, categoric_values):

    """
    It returns the Regression Table element of the model.

    Parameters
    ----------
    model :
        An Xgboost model instance.
    derived_col_names : List
        Contains column names after preprocessing.
    target_name : String
        Name of the Target column.
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    merge : List
        Nyoka's RegressionTable object

    """
    merge = list()
    if 'XGBClassifier' in str(model.__class__):
        if len(model.classes_) == 2:
            merge.append(
                pml.RegressionTable(
                    NumericPredictor=[pml.NumericPredictor(coefficient='1.0',name=feature_names[0])],
                    intercept='0.0',
                    targetCategory=str(model.classes_[-1])
                )
            )
            merge.append(
                pml.RegressionTable(intercept='0.0', targetCategory=str(model.classes_[0]))
            )
        else:
            for feat_idx in range(len(feature_names)):
                merge.append(
                    pml.RegressionTable(
                        NumericPredictor=[pml.NumericPredictor(coefficient='1.0',name=feature_names[feat_idx])],
                        intercept='0.0',
                        targetCategory=str(model.classes_[feat_idx])
                    )
                )
    return merge


def get_dtype(feat_value):
    """
    It return the data type of the value.

    Parameters
    ----------
    feat_value :
        Contains a value for finding the its data type.

    Returns
    -------
        Returns the respective data type of that value.

    """
    data_type=feat_value.__class__.__name__
    if 'float' in data_type:
        return DATATYPE.DOUBLE
    if 'int' in data_type:
        return DATATYPE.INTEGER
    if 'str' in data_type:
        return DATATYPE.STRING


def has_target(model):
    """
    Checks whether a given model has target or not

    Parameters
    ----------
    model :
        Scikit-learn's model object

    Returns
    -------
    Boolean value
    """
    target_less_models = ['OneClassSVM','IsolationForest', ]
    if model.__class__.__name__  in target_less_models:
        return False
    else:
        return True


def get_multiple_model_method(model):
    """
    It returns the type of multiple model method for MiningModels.

    Parameters
    ----------
    model :
        Contains Xgboost model object
    Returns
    -------
    The multiple model method for a MiningModel.

    """
    if 'XGBClassifier' in str(model.__class__):
        return MULTIPLE_MODEL_METHOD.MODEL_CHAIN
    else:
        return MULTIPLE_MODEL_METHOD.SUM

