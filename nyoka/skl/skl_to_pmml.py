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

import nyoka.PMML44 as pml
from nyoka.skl import pre_process as pp
from datetime import datetime
import math
from nyoka.base.constants import *

def skl_to_pmml(pipeline, col_names, target_name='target', pmml_f_name='from_sklearn.pmml', model_name=None, description=None):

    """
    Exports scikit-learn pipeline object into pmml

    Parameters
    ----------
    pipeline :
        Contains an instance of Pipeline with preprocessing and final estimator
    col_names : List
        Contains list of feature/column names.
    target_name : String
        Name of the target column. (Default='target')
    pmml_f_name : String
        Name of the pmml file. (Default='from_sklearn.pmml')
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description of the model

    Returns
    -------
    Generates a PMML object and exports it to `pmml_f_name` 
    
    """
    try:
        model = pipeline.steps[-1][1]
    except:
        raise TypeError("Exporter expects pipeleine_instance and not an estimator_instance")
    else:
        import numpy as np
        if isinstance(col_names, np.ndarray):
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
            Header=get_header(description),
            DataDictionary=get_data_dictionary(model, col_names, target_name, categoric_values),
            **trfm_dict_kwargs,
            **PMML_kwargs
        )
        pmml.export(outfile=open(pmml_f_name, "w"), level=0)


def any_in(seq_a, seq_b):
    """
    Checks for common elements in two given sequence elements

    Parameters
    ----------
    seq_a : list
        A list of items

    seq_b : list
        A list of items

    Returns
    -------
    Returns a boolean value if any item of seq_a belongs to seq_b or visa versa

    """
    return any(elem in seq_b for elem in seq_a)


def get_PMML_kwargs(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, model_name):

    """
    It returns all the pmml elements.

    Parameters
    ----------
    model : Scikit-learn model object
        An instance of Scikit-learn model.
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
    algo_kwargs : Dictionary
        Get the PMML model argument based on scikit learn model object
    """
    skl_mdl_super_cls_names = get_super_cls_names(model)
    regression_model_names = ('LinearRegression','LinearSVR')
    regression_mining_model_names = ('LogisticRegression', 'RidgeClassifier','LinearDiscriminantAnalysis', \
                                        'SGDClassifier','LinearSVC',)
    tree_model_names = ('BaseDecisionTree',)
    support_vector_model_names = ('SVC', 'SVR')
    anomaly_model_names = ('OneClassSVM','IsolationForest')
    naive_bayes_model_names = ('GaussianNB',)
    mining_model_names = ('RandomForestRegressor', 'RandomForestClassifier', 'GradientBoostingClassifier',
                            'GradientBoostingRegressor')
    neurl_netwk_model_names = ('MLPClassifier', 'MLPRegressor')
    nearest_neighbour_names = ('NeighborsBase',)
    clustering_model_names = ('KMeans',)
    if any_in(tree_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'TreeModel': get_tree_models(model,
                                                    derived_col_names,
                                                    col_names,
                                                    target_name,
                                                    mining_imp_val,
                                                    categoric_values,
                                                    model_name)}
    elif any_in(regression_mining_model_names, skl_mdl_super_cls_names):
        if len(model.classes_) == 2:
            algo_kwargs = {'RegressionModel': get_regrs_models(model,
                                                           derived_col_names,
                                                           col_names,
                                                           target_name,
                                                           mining_imp_val,
                                                           categoric_values,
                                                           model_name)}
        else:
            algo_kwargs = {'MiningModel': get_reg_mining_models(model,
                                                                derived_col_names,
                                                                col_names,
                                                                target_name,
                                                                mining_imp_val,
                                                                categoric_values,
                                                                model_name)}
    elif any_in(regression_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'RegressionModel': get_regrs_models(model,
                                                           derived_col_names,
                                                           col_names,
                                                           target_name,
                                                           mining_imp_val,
                                                           categoric_values,
                                                           model_name)}
    elif any_in(support_vector_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'SupportVectorMachineModel':
                           get_supportVectorMachine_models(model,
                                                           derived_col_names,
                                                           col_names,
                                                           target_name,
                                                           mining_imp_val,
                                                           categoric_values,
                                                           model_name)}
    elif any_in(mining_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'MiningModel': get_ensemble_models(model,
                                                          derived_col_names,
                                                          col_names,
                                                          target_name,
                                                          mining_imp_val,
                                                          categoric_values,
                                                          model_name)}
    elif any_in(neurl_netwk_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'NeuralNetwork': get_neural_models(model,
                                                          derived_col_names,
                                                          col_names,
                                                          target_name,
                                                          mining_imp_val,
                                                          categoric_values,
                                                          model_name)}
    elif any_in(naive_bayes_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'NaiveBayesModel': get_naiveBayesModel(model,
                                                              derived_col_names,
                                                              col_names,
                                                              target_name,
                                                              mining_imp_val,
                                                              categoric_values,
                                                              model_name)}
    elif any_in(nearest_neighbour_names, skl_mdl_super_cls_names):
        algo_kwargs = {'NearestNeighborModel':
                           get_nearestNeighbour_model(model,
                                                      derived_col_names,
                                                      col_names,
                                                      target_name,
                                                      mining_imp_val,
                                                      categoric_values,
                                                      model_name)}
    elif any_in(anomaly_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'AnomalyDetectionModel':
                            get_anomalydetection_model(model,
                                                        derived_col_names,
                                                        col_names,
                                                        target_name,
                                                        mining_imp_val,
                                                        categoric_values,
                                                        model_name)}
    elif any_in(clustering_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'ClusteringModel':
                            get_clustering_model(model,
                                                    derived_col_names,
                                                    col_names,
                                                    target_name,
                                                    mining_imp_val,
                                                    categoric_values,
                                                    model_name
                                                 )}
    else:
        raise NotImplementedError("{} is not Implemented!".format(model.__class__.__name__))

    return algo_kwargs


def get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values):

    """
    It returns all the model element for a specific model.

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
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


def get_reg_mining_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, model_name):
    """
    Creates xml elements for multi-class linear models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing
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
    mining_model : List
        Returns a Nyoka's MiningModel object

    """
    num_classes = len(model.classes_)
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values)

    mining_model = pml.MiningModel(modelName=model_name if model_name else model.__class__.__name__,**model_kwargs)
    inner_mining_schema = [mfield for mfield in model_kwargs['MiningSchema'].MiningField if mfield.usageType != FIELD_USAGE_TYPE.TARGET]
    segmentation = pml.Segmentation(multipleModelMethod=MULTIPLE_MODEL_METHOD.MODEL_CHAIN)
    for idx in range(num_classes):
        segment = pml.Segment(id=str(idx+1),True_=pml.True_())
        segment.RegressionModel = pml.RegressionModel(
            functionName=MINING_FUNCTION.REGRESSION,
            MiningSchema=pml.MiningSchema(
                MiningField=inner_mining_schema
                ),
            Output=pml.Output(
                OutputField=[
                    pml.OutputField(
                        name="probablity_"+str(idx),
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                        )
                    ]
                ),
            RegressionTable=get_reg_tab_for_reg_mining_model(model,derived_col_names,idx,categoric_values)
        )
        if model.__class__.__name__ != 'LinearSVC':
            segment.RegressionModel.normalizationMethod = REGRESSION_NORMALIZATION_METHOD.LOGISTIC
        segmentation.add_Segment(segment)

    last_segment = pml.Segment(id=str(num_classes+1),True_=pml.True_())
    mining_flds_for_last = [pml.MiningField(name="probablity_"+str(idx)) for idx in range(num_classes)]
    mining_flds_for_last.append(pml.MiningField(name=target_name,usageType=FIELD_USAGE_TYPE.TARGET))
    mining_schema_for_last = pml.MiningSchema(MiningField=mining_flds_for_last)
    reg_tab_for_last = list()
    for idx in range(num_classes):
        reg_tab_for_last.append(
            pml.RegressionTable(
                intercept="0.0",
                targetCategory=str(model.classes_[idx]),
                NumericPredictor=[pml.NumericPredictor(
                    name="probablity_"+str(idx),
                    coefficient="1.0"
                )]
            )
        )

    last_segment.RegressionModel = pml.RegressionModel(
        functionName=MINING_FUNCTION.CLASSIFICATION,
        MiningSchema=mining_schema_for_last,
        RegressionTable=reg_tab_for_last
    )
    if model.__class__.__name__ != 'LinearSVC':
        last_segment.RegressionModel.normalizationMethod = REGRESSION_NORMALIZATION_METHOD.SIMPLEMAX
    segmentation.add_Segment(last_segment)
    mining_model.set_Segmentation(segmentation)
    return [mining_model]


def get_reg_tab_for_reg_mining_model(model, col_names, index, categorical_values):
    """
    Generates Regression Table for multi-class linear models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    col_names : List
        Contains list of feature/column names.
    index : int
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    Returns Nyoka's RegressionTable object
    """
    reg_tab = pml.RegressionTable(intercept="{:.16f}".format(model.intercept_[index]))
    for idx, coef in enumerate(model.coef_[index]):
        reg_tab.add_NumericPredictor(pml.NumericPredictor(name=col_names[idx],coefficient="{:.16f}".format(coef)))
    return [reg_tab]


def get_anomalydetection_model(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, model_name):
    """
    Creates xml elements for anomaly detction models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing
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
    anomaly_detection_model : List
        Returns Nyoka's AnomalyDetectionModel object

    """
    anomaly_detection_model = list()
    if 'OneClassSVM' in str(model.__class__):
        svm_model = get_supportVectorMachine_models(model,
                                                    derived_col_names,
                                                    col_names,
                                                    target_name,
                                                    mining_imp_val,
                                                    categoric_values, model_name)[0]
        anomaly_detection_model.append(
            pml.AnomalyDetectionModel(
                modelName=model_name if model_name else model.__class__.__name__,
                algorithmType=ANOMALY_DETECTION_ALGORITHM.ONE_CLASS_SVM,
                functionName=MINING_FUNCTION.REGRESSION,
                MiningSchema=get_mining_schema(model, col_names, target_name, mining_imp_val,categoric_values),
                Output=get_anomaly_detection_output(model),
                SupportVectorMachineModel=svm_model
            )
        )
    else:
        mining_schema = get_mining_schema(model, col_names, target_name, mining_imp_val,categoric_values)
        ensemble_model = get_ensemble_models(model,
                                            derived_col_names,
                                            col_names,
                                            'avg_path_length',
                                            mining_imp_val,
                                            categoric_values, model_name)[0]
        anomaly_detection_model.append(
            pml.AnomalyDetectionModel(
                modelName=model_name if model_name else "IsolationForest",
                algorithmType=ANOMALY_DETECTION_ALGORITHM.ISOLATION_FOREST,
                functionName=MINING_FUNCTION.REGRESSION,
                MiningSchema=mining_schema,
                Output=get_anomaly_detection_output(model),
                sampleDataSize=str(model.max_samples_),
                MiningModel=ensemble_model
            )
    )
    return anomaly_detection_model


def get_anomaly_detection_output(model):
    """
    Generates output for anomaly detection models

    Parameters
    ----------
    model :
        Scikit-learn's model object

    Returns
    -------
    output_fields :
        Returns Nyoka's Output object
    """
    output_fields = list()
    output_fields.append(pml.OutputField(name="anomalyScore",
                                            optype=OPTYPE.CONTINUOUS,
                                            dataType=DATATYPE.DOUBLE,
                                            feature=RESULT_FEATURE.PREDICTED_VALUE,
                                            isFinalResult="false"))
    thresh = 0
    try:
        thresh = model.threshold_
    except:
        thresh = 0

    offset = 0
    operator = SIMPLE_PREDICATE_OPERATOR.LESS_THAN
    if model.__class__.__name__ == "IsolationForest":
        operator = SIMPLE_PREDICATE_OPERATOR.GREATER_THAN
        offset = model.offset_
    thresh = -1 * (thresh + offset)

    output_fields.append(
        pml.OutputField(name="outlier",
                        optype=OPTYPE.CATEGORICAL,
                        dataType=DATATYPE.BOOLEAN,
                        feature=RESULT_FEATURE.DECISION,
                        isFinalResult="true",
                        Apply=pml.Apply(function=operator,
                                        FieldRef=[pml.FieldRef(field="anomalyScore")],
                                        Constant=[pml.Constant(dataType=DATATYPE.DOUBLE,
                                        valueOf_="0" if thresh==0 else "{:.16f}".format(thresh))]))
    )
    return pml.Output(OutputField=output_fields)


def get_clustering_model(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):
    """
    Generates PMML elements for clustering models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing
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
    clustering_models : List
        Returns Nyoka's ClusteringModel object

    """
    import numpy as np
    clustering_models = list()
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    values, counts = np.unique(model.labels_,return_counts=True)
    model_kwargs["Output"] = get_output_for_clustering(values)
    clustering_models.append(
        pml.ClusteringModel(
            modelClass=CLUSTERING_MODEL_CLASS.CENTER_BASED,
            modelName=model_name if model_name else model.__class__.__name__,
            numberOfClusters=get_cluster_num(model),
            ComparisonMeasure=get_comp_measure(),
            ClusteringField=get_clustering_flds(derived_col_names),
            Cluster=get_cluster_vals(model,counts),
            **model_kwargs
        )
    )

    return clustering_models


def get_output_for_clustering(values):
    """
    Generates output for clustering models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------
    output_fields : List
        Returns Nyoka's Output object
    """
    output_fields = list()
    for idx, val in enumerate(values):
        output_fields.append(
            pml.OutputField(
                name="affinity("+str(idx)+")",
                optype=OPTYPE.CONTINUOUS,
                dataType=DATATYPE.DOUBLE,
                feature=RESULT_FEATURE.ENTITY_AFFINITY,
                value=str(val)
            )
        )
    output_fields.append(pml.OutputField(name="cluster", optype=OPTYPE.CATEGORICAL,\
        dataType=DATATYPE.STRING,feature=RESULT_FEATURE.PREDICTED_VALUE))
    return pml.Output(OutputField=output_fields)



def get_cluster_vals(model,counts):
    """
    Generates cluster information for clustering models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------
    cluster_flds : List
        Returns Nyoka's Cluster object

    """
    centroids = model.cluster_centers_
    cluster_flds = []
    for centroid_idx in range(centroids.shape[0]):
        centroid_values = ""
        centroid_flds = pml.ArrayType(type_=ARRAY_TYPE.REAL)
        for centroid_cordinate_idx in range(centroids.shape[1]):
            centroid_flds.content_[0].value = centroid_values + "{:.16f}".format(centroids[centroid_idx][centroid_cordinate_idx])
            centroid_values = centroid_flds.content_[0].value + " "
        cluster_flds.append(pml.Cluster(id=str(centroid_idx), Array=centroid_flds,size=str(counts[centroid_idx])))
    return cluster_flds


def get_cluster_num(model):
    """
    Returns number of cluster for clustering models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------

    model.n_clusters: Integer

        Returns the number of clusters

    """
    return model.n_clusters


def get_comp_measure():
    """
    Generates comparison measure information for clustering models

    Parameters
    ----------

    Returns
    -------
    Returns Nyoka's ComparisonMeasure object

    """
    comp_equation = pml.euclidean()
    return pml.ComparisonMeasure(euclidean=comp_equation, kind=COMPARISON_MEASURE_KIND.DISTANCE)


def get_clustering_flds(col_names):
    """
    Generates cluster fields for clustering models

    Parameters
    ----------
    col_names :
        Contains list of feature/column names.

    Returns
    -------
    clustering_flds: List
        Returns Nyoka's ClusteringField object

    """
    clustering_flds = []
    for name in col_names:
        clustering_flds.append(pml.ClusteringField(field=str(name)))
    return clustering_flds


def get_nearestNeighbour_model(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):

    """
    Generates PMML elements for nearest neighbour model

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing
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
    nearest_neighbour_model :
        Returns Nyoka's NearestNeighborModel object

    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    nearest_neighbour_model = list()
    nearest_neighbour_model.append(
        pml.NearestNeighborModel(
            modelName=model_name if model_name else model.__class__.__name__,
            continuousScoringMethod=CONTINUOUS_SCORING_METHOD.AVERAGE,
            algorithmName="KNN",
            numberOfNeighbors=model.n_neighbors,
            KNNInputs=get_knn_inputs(derived_col_names),
            ComparisonMeasure=get_comparison_measure(model),
            TrainingInstances=get_training_instances(model, derived_col_names, target_name),
            **model_kwargs
        )
    )
    return nearest_neighbour_model


def get_training_instances(model, derived_col_names, target_name):

    """
    It returns the Training Instance element.

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing
    target_name : String
        Name of the Target column.

    Returns
    -------
    TrainingInstances :
        Returns Nyoka's TrainingInstances object

    """
    return pml.TrainingInstances(
        InstanceFields=get_instance_fields(derived_col_names, target_name),
        InlineTable=get_inline_table(model)
    )


def get_inline_table(model):
    """
    It Returns the Inline Table element of the model.

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------
    InlineTable :
        Returns Nyoka's InlineTable object

    """
    rows = []
    x = model._tree.get_arrays()[0].tolist()
    y = model._y.tolist()

    X = []
    for idx in range(len(model._tree.get_arrays()[0][0])):
        X.append("x" + str(idx + 1))

    for idx in range(len(x)):
        row = pml.row()
        row.elementobjs_ = ['y'] + X
        if hasattr(model, 'classes_'):
            row.y = model.classes_[y[idx]]
        else:
            row.y = y[idx]
        for idx_2 in range(len(x[idx])):
            exec("row." + X[idx_2] + "=" + str(x[idx][idx_2]))
        rows.append(row)
    return pml.InlineTable(row=rows)


def get_instance_fields(derived_col_names, target_name):
    """
    It returns the Instance field element.

    Parameters
    ----------

    derived_col_names : List
        Contains column names after preprocessing.
    target_name : String
        Name of the Target column.


    Returns
    -------
    InstanceFields :
        Returns Nyoka's InstanceFields object

    """
    instance_fields = list()
    instance_fields.append(pml.InstanceField(field=target_name, column="y"))
    for (index, name) in enumerate(derived_col_names):
        instance_fields.append(pml.InstanceField(field=str(name), column="x" + str(index + 1)))
    return pml.InstanceFields(InstanceField=instance_fields)


def get_comparison_measure(model):

    """
    It return the Comparison measure element for nearest neighbour model.

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------
    comp_measure :
        Returns Nyoka's ComparisonMeasure object.

    """
    if model.effective_metric_ == 'euclidean':
        comp_measure = pml.ComparisonMeasure(euclidean=pml.euclidean(), kind=COMPARISON_MEASURE_KIND.DISTANCE)
    elif model.effective_metric_ == 'minkowski':
        comp_measure = pml.ComparisonMeasure(minkowski=pml.minkowski(p_parameter=model.p), kind=COMPARISON_MEASURE_KIND.DISTANCE)
    elif model.effective_metric_ in ['manhattan','cityblock']:
        comp_measure = pml.ComparisonMeasure(cityBlock=pml.cityBlock(), kind=COMPARISON_MEASURE_KIND.DISTANCE)
    elif model.effective_metric_ == 'sqeuclidean':
        comp_measure = pml.ComparisonMeasure(squaredEuclidean=pml.squaredEuclidean(), kind=COMPARISON_MEASURE_KIND.DISTANCE)
    elif model.effective_metric_ == 'chebyshev':
        comp_measure = pml.ComparisonMeasure(chebychev=pml.chebychev(), kind=COMPARISON_MEASURE_KIND.DISTANCE)
    elif model.effective_metric_ == 'matching':
        comp_measure = pml.ComparisonMeasure(simpleMatching=pml.simpleMatching(), kind=COMPARISON_MEASURE_KIND.SIMILARITY)
    elif model.effective_metric_ == 'jaccard':
        comp_measure = pml.ComparisonMeasure(jaccard=pml.jaccard(), kind=COMPARISON_MEASURE_KIND.SIMILARITY)
    elif model.effective_metric_ == 'rogerstanimoto':
        comp_measure = pml.ComparisonMeasure(tanimoto=pml.tanimoto(), kind=COMPARISON_MEASURE_KIND.SIMILARITY)
    else:
        raise NotImplementedError("{} metric is not implemented for KNN Model!".format(model.effective_metric_))
    return comp_measure


def get_knn_inputs(col_names):
    """
    It returns the KNN Inputs element.

    Parameters
    ----------
    col_names : List
        Contains list of feature/column names.

    Returns
    -------
    KNNInputs :
        Returns Nyoka's KNNInputs object.

    """
    knnInput = list()
    for name in col_names:
        knnInput.append(pml.KNNInput(field=str(name)))
    return pml.KNNInputs(KNNInput=knnInput)


def get_naiveBayesModel(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):

    """
    Generates PMML elements for naive bayes models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
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
    naive_bayes_model : List
        Returns Nyoka's NaiveBayesModel
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    naive_bayes_model = list()
    naive_bayes_model.append(pml.NaiveBayesModel(
        modelName=model_name if model_name else model.__class__.__name__,
        BayesInputs=get_bayes_inputs(model, derived_col_names),
        BayesOutput=get_bayes_output(model, target_name),
        threshold=get_threshold(),
        **model_kwargs
    ))
    return naive_bayes_model


def get_threshold():
    """
    It returns the Threshold value for Naive Bayes models.

    Returns
    -------
    Returns the Threshold value

    """
    return '0.001'


def get_bayes_output(model, target_name):

    """
    It returns the Bayes Output element of the model

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    target_name : String
        Name of the Target column.

    Returns
    -------
    BayesOutput :
        Returns Nyoka's BayesOutput object

    """
    class_counts = model.class_count_
    target_val_counts = pml.TargetValueCounts()
    for name, count in zip(model.classes_, class_counts):
        tr_val = pml.TargetValueCount(value=str(name), count=str(count))
        target_val_counts.add_TargetValueCount(tr_val)
    return pml.BayesOutput(
        fieldName=target_name,
        TargetValueCounts=target_val_counts
    )



def get_bayes_inputs(model, derived_col_names):

    """
    It returns the Bayes Input element of the naive bayes model .

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing.

    Returns
    -------
    bayes_inputs :
        Returns Nyoka's BayesInput object.

    """
    bayes_inputs = pml.BayesInputs()
    for indx, name in enumerate(derived_col_names):
        means = model.theta_[:, indx]
        variances = model.sigma_[:, indx]
        target_val_stats = pml.TargetValueStats()
        for idx, val in enumerate(model.classes_):
            target_val = pml.TargetValueStat(
                val, GaussianDistribution=pml.GaussianDistribution(
                    mean="{:.16f}".format(means[idx]),
                    variance="{:.16f}".format(variances[idx])))
            target_val_stats.add_TargetValueStat(target_val)
        bayes_inputs.add_BayesInput(pml.BayesInput(fieldName=str(name),
                                               TargetValueStats=target_val_stats))
    return bayes_inputs


def get_supportVectorMachine_models(model, derived_col_names, col_names, target_names,
 									mining_imp_val, categoric_values,model_name):

    """
    Generates PMML elements for support vector machine models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing.
    col_names : List
        Contains list of feature/column names.
    target_names : String
        Name of the Target column.
    mining_imp_val : tuple
        Contains the mining_attributes,mining_strategy, mining_impute_value
    categoric_values : tuple
        Contains Categorical attribute names and its values
    model_name : string
        Name of the model

    Returns
    -------
    supportVector_models : List
        Returns Nyoka's SupportVectorMachineModel object

    """
    model_kwargs = get_model_kwargs(model, col_names, target_names, mining_imp_val,categoric_values)
    supportVector_models = list()
    kernel_type = get_kernel_type(model)
    supportVector_models.append(pml.SupportVectorMachineModel(
        modelName=model_name if model_name else model.__class__.__name__,
        classificationMethod=get_classificationMethod(model),
        VectorDictionary=get_vectorDictionary(model, derived_col_names, categoric_values),
        SupportVectorMachine=get_supportVectorMachine(model),
        **kernel_type,
        **model_kwargs
    ))

    return supportVector_models


def get_ensemble_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,model_name):

    """
    Generates PMML elemenets for ensemble models

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
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
    mining_models : List
        Returns Nyoka's MiningModel object
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    if model.__class__.__name__ == 'GradientBoostingRegressor':
        model_kwargs['Targets'] = get_targets(model, target_name)

    mining_models = list()
    mining_models.append(pml.MiningModel(
        algorithmName='randomForest' if model.__class__.__name__ in ['RandomForestClassifier','RandomForestRegressor'] else None,
        modelName=model_name if model_name else model.__class__.__name__,
        Segmentation=get_outer_segmentation(model, derived_col_names, col_names, target_name,
                                            mining_imp_val, categoric_values, model_name),
        **model_kwargs
    ))
    return mining_models


def get_targets(model, target_name):

    """
    It returns the Target element of the model.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
    target_name : String
        Name of the Target column.

    Returns
    -------
    targets :
        Returns Nyoka's Target object
    """
    if model.__class__.__name__ == 'GradientBoostingRegressor':
        targets = pml.Targets(
            Target=[
                pml.Target(
                    field=target_name,
                    rescaleConstant="{:.16f}".format(model.init_.mean if hasattr(model.init_,"mean")
                                                     else model.init_.constant_.ravel()[0]),
                    rescaleFactor="{:.16f}".format(model.learning_rate)
                )
            ]
        )
    else:
        targets = pml.Targets(
            Target=[
                pml.Target(
                    field=target_name,
                    rescaleConstant="{:.16f}".format(model.base_score)
                )
            ]
        )
    return targets


def get_multiple_model_method(model):

    """
    It returns the type of multiple model method for MiningModels.

    Parameters
    ----------
    model :
        A Scikit-learn model instance

    Returns
    -------
    The multiple model method for a MiningModel.

    """
    if model.__class__.__name__ == 'GradientBoostingClassifier':
        return MULTIPLE_MODEL_METHOD.MODEL_CHAIN
    elif model.__class__.__name__ == 'GradientBoostingRegressor':
        return MULTIPLE_MODEL_METHOD.SUM
    elif model.__class__.__name__ in ['RandomForestRegressor','IsolationForest','RandomForestClassifier']:
        return MULTIPLE_MODEL_METHOD.AVERAGE


def get_outer_segmentation(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,model_name):

    """
    It returns the Segmentation element of a MiningModel.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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
        Nyoka's Segmentation object

    """
    segmentation = pml.Segmentation(
        multipleModelMethod=get_multiple_model_method(model),
        Segment=get_segments(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,model_name)
    )
    return segmentation


def get_segments(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,model_name):

    """
    It returns the Segment element of a Segmentation.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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
    segments :
        Nyoka's Segment object

    """
    segments = None
    if 'GradientBoostingClassifier' in str(model.__class__):
        segments = get_segments_for_gbc(model, derived_col_names, col_names, target_name,
                                        mining_imp_val, categoric_values, model_name)
    else:
        segments = get_inner_segments(model, derived_col_names, col_names, 0)
    return segments


def get_segments_for_gbc(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, model_name):

    """
    It returns list of Segments element of a Segmentation.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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
    segments : List
        Nyoka's Segment object

    """
    import numpy as np
    segments = list()
    out_field_names = list()
    for estm_idx in range(len(model.estimators_[0])):
        mining_fields_for_first = list()
        for name in col_names:
            mining_fields_for_first.append(pml.MiningField(name=name))

        miningschema_for_first = pml.MiningSchema(MiningField=mining_fields_for_first)
        output_fields = list()
        output_fields.append(
            pml.OutputField(
                name='decisionFunction(' + str(estm_idx) + ')',
                feature=RESULT_FEATURE.PREDICTED_VALUE,
                dataType=DATATYPE.DOUBLE,
                isFinalResult=False
            )
        )
        if len(model.classes_) == 2:
            if hasattr(model.init_,"prior"):
                const = float(model.init_.prior)
            else:
                proba_pos_class = model.init_.class_prior_[1]
                eps = np.finfo(np.float32).eps
                proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
                const = np.log(proba_pos_class / (1 - proba_pos_class))
            output_fields.append(
                pml.OutputField(
                    name='transformedDecisionFunction(0)',
                    feature=RESULT_FEATURE.TRANSFORMED_VALUE,
                    dataType=DATATYPE.DOUBLE,
                    isFinalResult=True,
                    Apply=pml.Apply(function="+",
                        Apply_member=[pml.Apply(
                                function=FUNCTION.MULTIPLICATION,
                                Constant=[pml.Constant(
                                    dataType=DATATYPE.DOUBLE,
                                    valueOf_="{:.16f}".format(model.learning_rate)
                                )],
                                FieldRef=[pml.FieldRef(
                                    field="decisionFunction(0)",
                            )]
                        )],
                        Constant=[pml.Constant(valueOf_=const)]
                    )
                )
            )
        else:
            if hasattr(model.init_,"priors"):
                const = model.init_.priors
            else:
                probas = model.init_.class_prior_
                eps = np.finfo(np.float32).eps
                probas = np.clip(probas, eps, 1 - eps)
                const=np.log(probas).astype(np.float64)
            output_fields.append(
                pml.OutputField(
                    name='transformedDecisionFunction(' + str(estm_idx) + ')',
                    feature=RESULT_FEATURE.TRANSFORMED_VALUE,
                    dataType=DATATYPE.DOUBLE,
                    isFinalResult=True,
                    Apply=pml.Apply(function="+",
                        Apply_member=[pml.Apply(
                            function=FUNCTION.MULTIPLICATION,
                            Constant=[pml.Constant(
                                dataType=DATATYPE.DOUBLE,
                                valueOf_="{:.16f}".format(model.learning_rate)
                            )],
                            FieldRef=[pml.FieldRef(
                                field="decisionFunction(" + str(estm_idx) + ")",
                            )]
                        )],
                        Constant=[pml.Constant(valueOf_=const[estm_idx])]
                    )
                )
            )

        out_field_names.append('transformedDecisionFunction(' + str(estm_idx) + ')')
        segments.append(
            pml.Segment(
                True_=pml.True_(),
                id=str(estm_idx),
                MiningModel=pml.MiningModel(
                    functionName=MINING_FUNCTION.REGRESSION,
                    modelName="MiningModel",
                    MiningSchema=miningschema_for_first,
                    Output=pml.Output(OutputField=output_fields),
                    Segmentation=pml.Segmentation(
                        multipleModelMethod=MULTIPLE_MODEL_METHOD.SUM,
                        Segment=get_inner_segments(model, derived_col_names,
                                                   col_names, estm_idx)
                    )
                )
            )
        )
    reg_model = get_regrs_models(model, out_field_names,out_field_names, target_name, mining_imp_val, categoric_values, model_name)[0]
    reg_model.Output = None
    if len(model.classes_) == 2:
        reg_model.normalizationMethod=REGRESSION_NORMALIZATION_METHOD.LOGISTIC
    else:
        reg_model.normalizationMethod=REGRESSION_NORMALIZATION_METHOD.SOFTMAX
    segments.append(
        pml.Segment(
            id=str(len(model.estimators_[0])),
            True_=pml.True_(),
            RegressionModel=reg_model
        )
    )
    return segments


def get_inner_segments(model, derived_col_names, col_names, index):

    """
    It returns the segments of a Segmentation.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
    derived_col_names : List
        Contains column names after preprocessing.
    col_names : List
        Contains list of feature/column names.
    index : Integer
        The index of the estimator for the model

    Returns
    -------
    segments : List
        Nyoka's Segment object

    """
    import numpy as np
    segments = list()
    for estm_idx in range(model.n_estimators):
        if np.asanyarray(model.estimators_).ndim == 1:
            estm = model.estimators_[estm_idx]
        else:
            estm = model.estimators_[estm_idx][index]
        features_ = set(estm.tree_.feature)
        features_.discard(-2)
        if len(features_) != 0:
            nodes = get_node(estm, derived_col_names, model)
        else:
            nodes = pml.Node(
                True_=pml.True_(),
                id="0",
                score=estm.tree_.value.ravel()[0],
                recordCount=estm.tree_.n_node_samples[0]
            )
        mining_fields = list()
        if model.__class__.__name__ in ['RandomForestClassifier','RandomForestRegressor']:
            col_names = derived_col_names
        for feat in col_names:
            mining_fields.append(pml.MiningField(name=feat))
        segments.append(
            pml.Segment(
                True_=pml.True_(),
                id=str(estm_idx),
                TreeModel=pml.TreeModel(
                    modelName=estm.__class__.__name__,
                    functionName=get_mining_func(estm),
                    splitCharacteristic=TREE_SPLIT_CHARACTERISTIC.MULTI,
                    MiningSchema=pml.MiningSchema(MiningField = mining_fields),
                    Node=nodes
                )
            )
        )
    return segments


def get_classificationMethod(model):

    """
    It returns the Classification method name for SVM models.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.

    Returns
    -------
    Returns the classification method of the SVM model

    """
    if model.__class__.__name__ == 'SVC':
        return SVM_CLASSIFICATION_METHOD.OVO
    else:
        return SVM_CLASSIFICATION_METHOD.OVR


def get_vectorDictionary(model, derived_col_names, categoric_values):

    """
    It return the Vector Dictionary element.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
    derived_col_names : List
        Contains column names after preprocessing.
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    VectorDictionary :
        Nyoka's VectorDictionary object

    """
    fieldref_element = list()
    for name in derived_col_names:
        fieldref_element.append(pml.FieldRef(field=name))

    vectorfields_element = pml.VectorFields(FieldRef=fieldref_element)
    vec_id = list(model.support_)
    vecinsts = list()
    vecs = list(model.support_vectors_)
    if model.support_vectors_.__class__.__name__ != 'csr_matrix':
        for vec_idx in range(len(vecs)):
            vecinsts.append(pml.VectorInstance(
                id=vec_id[vec_idx],
                REAL_SparseArray=pml.REAL_SparseArray(
                    n=len(fieldref_element),
                    Indices=([x for x in range(1, len(vecs[vec_idx]) + 1)]),
                    REAL_Entries=vecs[vec_idx].tolist()
                )
            ))
    else:
        for vec_idx in range(len(vecs)):
            vecinsts.append(pml.VectorInstance(
                id=vec_id[vec_idx],
                REAL_SparseArray=pml.REAL_SparseArray(
                    n=len(fieldref_element),
                    Indices=([x for x in range(1, len(vecs[vec_idx].todense().tolist()[0]) + 1)]),
                    REAL_Entries=vecs[vec_idx].todense().tolist()[0]
                )
            ))
    vd=pml.VectorDictionary(VectorFields=vectorfields_element, VectorInstance=vecinsts)
    return vd


def get_kernel_type(model):

    """
    It returns the kernel type element.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.

    Returns
    -------
    kernel_kwargs : Dictionary
        Get the respective kernel type of the SVM model.

    """
    kernel_kwargs = dict()
    if model.kernel == 'linear':
        kernel_kwargs['LinearKernelType'] = pml.LinearKernelType(description='Linear Kernel Type')
    elif model.kernel == 'poly':
        kernel_kwargs['PolynomialKernelType'] = pml.PolynomialKernelType(description='Polynomial Kernel type',
                                                                         gamma="{:.16f}".format(model._gamma),
                                                                         coef0="{:.16f}".format(model.coef0),
                                                                         degree=model.degree)
    elif model.kernel == 'rbf':
        kernel_kwargs['RadialBasisKernelType'] = pml.RadialBasisKernelType(description='Radial Basis Kernel Type',
                                                                           gamma="{:.16f}".format(model._gamma))
    elif model.kernel == 'sigmoid':
        kernel_kwargs['SigmoidKernelType'] = pml.SigmoidKernelType(description='Sigmoid Kernel Type',
                                                               gamma="{:.16f}".format(model._gamma),
                                                               coef0="{:.16f}".format(model.coef0))
    else:
        raise NotImplementedError("{} kernel is not implemented!".format(model.kernel))
    return kernel_kwargs


def get_supportVectorMachine(model):

    """
    Generates PMML elements for support vector machine models

    Parameters
    ----------
    model :
        A Scikit-learn model instance.

    Returns
    -------
    support_vector_machines : List
        Nyoka's SupportVectorMachineModel object

    """
    support_vector_machines = list()
    if model.__class__.__name__ in ['SVR','OneClassSVM']:
        support_vector = list()
        for sv in model.support_:
            support_vector.append(pml.SupportVector(vectorId=sv))
        support_vectors = pml.SupportVectors(SupportVector=support_vector)
        coefficient = list()
        absoValue = model.intercept_[0]
        if model.dual_coef_.__class__.__name__ != 'csr_matrix':
            for coef in model.dual_coef_:
                for num in coef:
                    coefficient.append(pml.Coefficient(value="{:.16f}".format(num)))
        else:
            dual_coefficent=model.dual_coef_.data
            for num in dual_coefficent:
                coefficient.append(pml.Coefficient(value="{:.16f}".format(num)))
        coeff = pml.Coefficients(absoluteValue=absoValue, Coefficient=coefficient)
        support_vector_machines.append(pml.SupportVectorMachine(SupportVectors=support_vectors, Coefficients=coeff))
    else:
        import numpy as np
        support_vector_locs = np.cumsum(np.hstack([[0], model.n_support_]))
        n_class = model.dual_coef_.shape[0] + 1
        coef_abs_val_index = 0
        for class1 in range(n_class):
            sv1 = model.support_[support_vector_locs[class1]:support_vector_locs[class1 + 1]]
            for class2 in range(class1 + 1, n_class):
                svs = list()
                coefs = list()
                sv2 = model.support_[support_vector_locs[class2]:support_vector_locs[class2 + 1]]
                svs.append((list(sv1) + list(sv2)))
                alpha1 = model.dual_coef_[class2 - 1, support_vector_locs[class1]:support_vector_locs[class1 + 1]]
                alpha2 = model.dual_coef_[class1, support_vector_locs[class2]:support_vector_locs[class2 + 1]]
                coefs.append((list(alpha1) + list(alpha2)))
                all_svs = list()
                for sv in (svs[0]):
                    all_svs.append(pml.SupportVector(vectorId=sv))
                all_coefs = list()
                for coef in (coefs[0]):
                    all_coefs.append(pml.Coefficient(value="{:.16f}".format(coef)))
                coef_abs_value = model.intercept_[coef_abs_val_index]
                coef_abs_val_index += 1
                if len(model.classes_) == 2:
                    support_vector_machines.append(
                        pml.SupportVectorMachine(
                            targetCategory=model.classes_[class1],
                            alternateTargetCategory=model.classes_[class2],
                            SupportVectors=pml.SupportVectors(SupportVector=all_svs),
                            Coefficients=pml.Coefficients(absoluteValue="{:.16f}".format(coef_abs_value), Coefficient=all_coefs)
                        )
                    )
                else:
                    support_vector_machines.append(
                        pml.SupportVectorMachine(
                            targetCategory=model.classes_[class2],
                            alternateTargetCategory=model.classes_[class1],
                            SupportVectors=pml.SupportVectors(SupportVector=all_svs),
                            Coefficients=pml.Coefficients(absoluteValue="{:.16f}".format(coef_abs_value), Coefficient=all_coefs)
                        )
                    )
    return support_vector_machines


def get_tree_models(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,model_name):

    """
    Generates PMML elements for tree models

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
    derived_col_names :
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
    tree_models : List
        Nyoka's TreeModel object

    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    tree_models = list()
    tree_models.append(pml.TreeModel(
        modelName=model_name if model_name else model.__class__.__name__,
        Node=get_node(model, derived_col_names),
        **model_kwargs
    ))
    return tree_models


def get_neural_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,model_name):

    """
    Generates PMML elements for neural network models

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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
    neural_model : List
        Nyoka's NeuralNetwork object

    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    neural_model = list()
    neural_layers, neural_outs = get_neural_layer(model, derived_col_names, target_name)
    neural_model.append(pml.NeuralNetwork(
        modelName=model_name if model_name else model.__class__.__name__,
        threshold='0',
        altitude='1.0',
        activationFunction=get_funct(model),
        NeuralInputs = get_neuron_input(derived_col_names),
        NeuralLayer = neural_layers,
        NeuralOutputs = neural_outs,
        **model_kwargs
    ))
    return neural_model


def get_funct(sk_model):

    """
    It returns the activation fucntion for a neural network model.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.

    Returns
    -------
    a_fn : String
        Returns the activation function.

    """
    a_fn = sk_model.activation
    if a_fn =='relu':
        a_fn = NN_ACTIVATION_FUNCTION.RECTIFIER
    return a_fn


def get_regrs_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,model_name):

    """
    Generates PMML elements for linear models

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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


def get_regrs_tabl(model, feature_names, target_name, categoric_values):

    """
    It returns the Regression Table element of the model.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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
    if hasattr(model, 'intercept_'):
        import numpy as np
        func_name = get_mining_func(model)
        inter = model.intercept_
        model_coef = model.coef_
        target_classes = target_name
        if not hasattr(inter, '__iter__') or model.__class__.__name__ in ['LinearRegression','LinearSVR']:
            inter = np.array([inter])
            target_classes = [target_classes]
            model_coef = np.ravel(model_coef)
            model_coef = model_coef.reshape(1, model_coef.shape[0])
            target_cat = None
        else:
            target_classes = model.classes_
            max_target_index = len(target_classes) - 1
            target_cat = target_classes[max_target_index]

        if hasattr(model_coef[0],"__len__"):
            model_coef = model_coef[0]
        reg_preds=list()
        for idx, feat in enumerate(feature_names):
            reg_preds.append(pml.NumericPredictor(name=feat, coefficient="{:.16f}".format(model_coef[idx])))
        merge.append(
            pml.RegressionTable(
                intercept="{:.16f}".format(inter.item()),
                targetCategory=target_cat,
                NumericPredictor=reg_preds
            )
        )
        if func_name != MINING_FUNCTION.REGRESSION:
            merge.append(
                pml.RegressionTable(
                    intercept="0.0",
                    targetCategory=target_classes[0]
                )
            )

    else:
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



def get_node(model, features_names, main_model=None):

    """
    It return the Node element of the model.

    Parameters
    ----------
    model :
        An instance of the estimator of the tree object.
    features_names : List
        Contains the list of feature/column name.
    main_model :
        A Scikit-learn model instance.

    Returns
    -------
    Get all the underlying Nodes.

    """
    tree = model.tree_
    node_samples = tree.n_node_samples
    if main_model and main_model.__class__.__name__ == 'RandomForestClassifier':
        classes = main_model.classes_
    elif hasattr(model,'classes_'):
        classes = model.classes_
    tree_leaf = -1

    def _getNode(idx,parent=None, cond=None):
        simple_pred_cond = None
        if cond:
            simple_pred_cond = cond
        node = pml.Node(id=idx, recordCount=float(tree.n_node_samples[idx]))
        if simple_pred_cond:
            node.SimplePredicate = simple_pred_cond
        else:
            node.True_ = pml.True_()


        if tree.children_left[idx] != tree_leaf:
            fieldName = features_names[tree.feature[idx]]
            prnt = None
            if model.__class__.__name__ == "ExtraTreeRegressor":
                prnt = parent + 1
            thresh = tree.threshold[idx].astype("float32")
            simplePredicate = pml.SimplePredicate(field=fieldName, operator=SIMPLE_PREDICATE_OPERATOR.LESS_OR_EQUAL,\
                value = str(thresh))
                                                #   value="{:.16f}".format(tree.threshold[idx]))
            left_child = _getNode(tree.children_left[idx],prnt, simplePredicate)
            simplePredicate = pml.SimplePredicate(field=fieldName, operator=SIMPLE_PREDICATE_OPERATOR.GREATER_THAN, \
                value= str(thresh))
                                                #   value="{:.16f}".format(tree.threshold[idx]))
            right_child = _getNode(tree.children_right[idx],prnt, simplePredicate)
            node.add_Node(left_child)
            node.add_Node(right_child)
        else:
            nodeValue = list(tree.value[idx][0])
            lSum = float(sum(nodeValue))
            if model.__class__.__name__ == 'DecisionTreeClassifier':
                probs = [x / lSum for x in nodeValue]
                score_dst = []
                for i in range(len(probs)):
                    score_dst.append(pml.ScoreDistribution(confidence=probs[i], recordCount=float(nodeValue[i]),
                                                          value=classes[i]))
                node.ScoreDistribution = score_dst
                node.score = classes[probs.index(max(probs))]
            else:
                if model.__class__.__name__ == "ExtraTreeRegressor":
                    nd_sam=node_samples[int(idx)]
                    node.score = "{:.16f}".format(parent+avgPathLength(nd_sam))
                else:
                    node.score="{:.16f}".format(lSum)
        return node
    if model.__class__.__name__ == "ExtraTreeRegressor":
        return _getNode(0,0)
    else:
        return _getNode(0)

def avgPathLength(n):
    """
    Generates average path length for Isolation forest models

    Parameters
    ----------
    n : int
        Number of samples

    Returns
    -------
    The average path length
    """
    if n<=1.0:
        return 1.0
    return 2.0*(math.log(n-1.0)+0.57721566) - 2.0*((n-1.0)/n)


def get_output(model, target_name):

    """
    It returns the output element of the model.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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




def get_mining_func(model):
    """
    It returns the name of the mining function of the model.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.

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
    # import numpy as np
    # if not hasattr(model, 'n_classes_'):
    #     if hasattr(model,'n_clusters'):
    #         func_name = MINING_FUNCTION.CLUSTERING
    #     else:
    #         func_name = MINING_FUNCTION.REGRESSION
    # else:
    #     # if isinstance(model.classes_, np.ndarray):
    #     if model.n_classes_ > 1:
    #         func_name = MINING_FUNCTION.CLASSIFICATION
    #     else:
    #         func_name = MINING_FUNCTION.REGRESSION

    return func_name


def get_mining_schema(model, feature_names, target_name, mining_imp_val, categoric_values):

    """
    It returns the Mining Schema of the model.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
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


def get_neuron_input(feature_names):

    """
    It returns the Neural Input element.

    Parameters
    ----------
    feature_names : List
        Contains the list of feature/column name.

    Returns
    -------
    neural_input_element :
        Returns Nyoka's NeuralInput object

    """
    neural_input = list()
    for features in feature_names:
        field_ref = pml.FieldRef(field = str(features))
        derived_flds = pml.DerivedField(optype = OPTYPE.CONTINUOUS, dataType = DATATYPE.DOUBLE, FieldRef = field_ref)
        class_node = pml.NeuralInput(id = str(features), DerivedField = derived_flds)
        neural_input.append(class_node)
    neural_input_element = pml.NeuralInputs(NeuralInput = neural_input, numberOfInputs = str(len(neural_input)))
    return neural_input_element


def get_neural_layer(model, feature_names, target_name):

    """
    It returns the Neural Layer and Neural Ouptput element.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
    feature_names : List
        Contains the list of feature/column name.
    target_name : String
        Name of the Target column.

    Returns
    -------
    all_neuron_layer : List
        Nyoka's NeuralLayer object

    neural_output_element :
        Nyoka's NeuralOutput object

    """
    weight = model.coefs_
    bias = model.intercepts_
    last_layer = bias[-1]
    hidden_layer_sizes = model.hidden_layer_sizes
    hidden_layers = list(hidden_layer_sizes)
    hidden_layers.append(len(last_layer))
    neuron = list()
    all_neuron_layer = list()
    input_features = feature_names
    neuron_id = list()
    for count in range(len(hidden_layers)):
        for count1 in range(hidden_layers[count]):
            con = list()
            for count2 in range(len(input_features)):
                con.append(pml.Con(from_ = input_features[count2], weight = format(weight[count][count2][count1])))
            neuron.append(pml.Neuron(id = str(count)+str(count1), bias = format(bias[count][count1]),Con = con))
            neuron_id.append(str(count)+str(count1))
        all_neuron_layer.append(pml.NeuralLayer(Neuron = neuron))
        input_features = neuron_id
        neuron_id = list()
        neuron = list()
    all_neuron_layer[-1].activationFunction = NN_ACTIVATION_FUNCTION.IDENTITY
    if hasattr(model, "classes_"):
        if len(model.classes_) == 2:
            bias1=[1.0,0.0]
            weight1=[-1.0,1.0]
            con = list()
            linear = ['logistic/1']
            i_d = ['false', 'true']
            con.append(pml.Con(from_ = input_features[0], weight = 1.0))
            neuron.append(pml.Neuron(id = linear[0], bias = ('0.0'), Con = con))
            all_neuron_layer.append(pml.NeuralLayer(activationFunction = NN_ACTIVATION_FUNCTION.LOGISTIC, Neuron = neuron))
            neuron = list()
            con = list()
            for num in range(2):
                con.append(pml.Con(from_ = linear[0], weight = format(weight1[num])))
                neuron.append(pml.Neuron(id = i_d[num], bias = format(bias1[num]), Con = con))
                con = list()
            all_neuron_layer.append(pml.NeuralLayer(activationFunction = NN_ACTIVATION_FUNCTION.IDENTITY, Neuron = neuron))
            input_features = i_d
        else:
            all_neuron_layer[-1].normalizationMethod = model.out_activation_


        neural_output = list()
        for values, count in zip(model.classes_, range(len(model.classes_))):
            norm_discrete = pml.NormDiscrete(field = target_name, value = str(values))
            derived_flds = pml.DerivedField(optype = OPTYPE.CATEGORICAL, dataType = DATATYPE.DOUBLE,
                                    NormDiscrete = norm_discrete)
            if len(input_features)==1:
                class_node = pml.NeuralOutput(outputNeuron = input_features[0], DerivedField = derived_flds)
            else:
                class_node = pml.NeuralOutput(outputNeuron = input_features[count],DerivedField = derived_flds)
            neural_output.append(class_node)
        neural_output_element = pml.NeuralOutputs(numberOfOutputs = None, Extension = None,
                                                    NeuralOutput = neural_output)
    else:
        neural_output = list()
        fieldRef = pml.FieldRef(field = target_name)
        derived_flds = pml.DerivedField(optype = OPTYPE.CONTINUOUS, dataType = DATATYPE.DOUBLE, FieldRef = fieldRef)
        class_node = pml.NeuralOutput(outputNeuron = input_features[0], DerivedField = derived_flds)
        neural_output.append(class_node)
        neural_output_element = pml.NeuralOutputs(numberOfOutputs = None, Extension = None, NeuralOutput = neural_output)

    return all_neuron_layer, neural_output_element


def get_super_cls_names(model_inst):
    """
    It returns the set of Super class of the model.

    Parameters
    -------
    model_inst :
        Instance of the scikit-learn model

    Returns
    -------
    parents : Set
        Returns all the parent class of the model instance.

    """
    def super_cls_names(cls):
        nonlocal parents
        parents.add(cls.__name__)
        for super_cls in cls.__bases__:
            super_cls_names(super_cls)
    cls = model_inst.__class__
    parents = set()
    super_cls_names(cls)
    return parents

def get_header(description):

    """
    It returns the Header element of the pmml.

     Returns
     -------
     header :
         Returns Nyoka's Header object.

     """
    copyryt = HEADER_INFO.COPYRIGHT
    description = description if description else HEADER_INFO.DEFAULT_DESCRIPTION
    timestamp = pml.Timestamp(datetime.now())
    application=pml.Application(name=HEADER_INFO.APPLICATION_NAME,version=HEADER_INFO.APPLICATION_VERSION)
    header = pml.Header(copyright=copyryt, description=description, Timestamp=timestamp, Application=application)
    return header


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

def get_data_dictionary(model, feature_names, target_name, categoric_values):

    """
    It returns the Data Dictionary element.

    Parameters
    ----------
    model :
        A Scikit-learn model instance.
    feature_names : List
        Contains the list of feature/column name.
    target_name : List
        Name of the Target column.
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    data_dict :
        Returns Nyoka's DataDictionary object

    """
    categoric_feature_name = list()
    if categoric_values:
        categoric_labels = categoric_values[0]
        categoric_feature_name = categoric_values[1]
    target_attr_values = []
    n_features = len(feature_names)
    features_pmml_optype = [OPTYPE.CONTINUOUS] * n_features
    features_pmml_dtype = [DATATYPE.DOUBLE] * n_features

    mining_func = get_mining_func(model)

    if mining_func == MINING_FUNCTION.CLASSIFICATION:
        target_pmml_optype = OPTYPE.CATEGORICAL
        target_pmml_dtype = get_dtype(model.classes_[0])
        target_attr_values = model.classes_.tolist()
    elif mining_func == MINING_FUNCTION.REGRESSION:
        target_pmml_optype = OPTYPE.CONTINUOUS
        target_pmml_dtype = DATATYPE.DOUBLE

    data_fields = list()
    if categoric_values:
        for class_list, attr_for_class in zip(categoric_labels, categoric_feature_name):
            category_flds = pml.DataField(name=str(attr_for_class), optype=OPTYPE.CATEGORICAL,
                                          dataType=get_dtype(class_list[0]) if class_list else DATATYPE.STRING)
            if class_list:
                for values in class_list:
                    category_flds.add_Value(pml.Value(value=str(values)))
            data_fields.append(category_flds)
    attr_without_class_attr = [feat_name for feat_name in feature_names if feat_name not in categoric_feature_name]
    for feature_idx, feat_name in enumerate(attr_without_class_attr):
        data_fields.append(pml.DataField(name=str(feat_name),
                                         optype=features_pmml_optype[feature_idx],
                                         dataType=features_pmml_dtype[feature_idx]))
    if model.__class__.__name__ not in ['KMeans', 'IsolationForest', 'OneClassSVM']:
        class_node = pml.DataField(name=str(target_name), optype=target_pmml_optype,
                                dataType=target_pmml_dtype)

        for class_value in target_attr_values:
            class_node.add_Value(pml.Value(value=str(class_value)))
        data_fields.append(class_node)
    data_dict = pml.DataDictionary(numberOfFields=len(data_fields), DataField=data_fields)
    return data_dict


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

