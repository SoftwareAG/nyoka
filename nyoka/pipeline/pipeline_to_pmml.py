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
import nyoka.xgboost.xgboost_to_pmml as xgbToPmml
from skl import pre_process as pp
from datetime import datetime
from base.constants import *


def pipeline_to_pmml(pipeline, col_names, target_name, pmml_f_name='from_pipeline.pmml',model_name=None,description=None):
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
            DataDictionary=sklToPmml.get_data_dictionary(model.final_estimators, col_names, target_name, categoric_values),
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
    algo_kwargs = {'MiningModel': get_estimator_models(model,
                                                      derived_col_names,
                                                      col_names,
                                                      target_name,
                                                      mining_imp_val,
                                                      categoric_values,
                                                      model_name)}
    return algo_kwargs


def get_estimator_models(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):
    model_segments = list()
    base_estimators = model.base_estimators
    i=0
    for i, estimator in enumerate(base_estimators):
        mining_model = xgbToPmml.get_ensemble_models(estimator, derived_col_names, col_names, target_name+"_"+str(i), mining_imp_val,categoric_values,model_name)[0]
        model_segments.append(pml.Segment(True_=pml.True_(), id=i+1, MiningModel=mining_model))
    i+=1
    final_col_names = model.final_estimators._Booster.feature_names
    mining_model = xgbToPmml.get_ensemble_models(model.final_estimators, derived_col_names, final_col_names, target_name, mining_imp_val,categoric_values,model_name)[0]
    model_segments.append(pml.Segment(True_=pml.True_(), id=i+1, MiningModel=mining_model))
    segmentation = pml.Segmentation(
            multipleModelMethod="modelChain",
            Segment=model_segments
        )
    model_kwargs = xgbToPmml.get_model_kwargs(model.final_estimators, col_names, target_name, mining_imp_val, categoric_values)
    mining_models = list()
    mining_models.append(pml.MiningModel(
        modelName=model_name if model_name else "Pipeline",
        Segmentation=segmentation,
        **model_kwargs
    ))
    return mining_models