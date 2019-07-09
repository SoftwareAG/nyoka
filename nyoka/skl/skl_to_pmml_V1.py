from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import PMML43Ext as pml
from skl import pre_process as pp
from datetime import datetime
import math
import metadata
import inspect
from nyoka.keras.keras_model_to_pmml import KerasToPmml
from nyoka.xgboost.xgboost_to_pmml import xgboost_to_pmml
from nyoka.lgbm.lgb_to_pmml import lgb_to_pmml
from nyoka.lgbm.lgbmTrainingAPI_to_pmml import ExportToPMML as ext

def model_to_pmml(toExportDict, pmml_f_name='from_sklearn.pmml'):

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

    Returns
    -------
    Returns a pmml file 
    
    """
    # To support multiple models and Transformation dictionaries
    models_dict = {'DeepNetwork':[]}
    trfm_dict_kwargs = {'TransformationDictionary':[]}
    data_dicts = []
    visited = []
    categoric_values = None
    derived_col_names = None
    mining_imp_val = None


    for model_name in toExportDict.keys():
        col_names = toExportDict[model_name]['featuresUsed']
        target_name = toExportDict[model_name]['targetName']
        tasktype = toExportDict[model_name]['taskType']

        model = toExportDict[model_name]['modelObj']

        pipelineOnly = toExportDict[model_name]['pipelineObj']

        categoric_values = tuple()
        derived_col_names = col_names
        mining_imp_val = tuple()

        if (pipelineOnly is not None) and (pipelineOnly not in visited):
            derived_col_names,categoric_values,mining_imp_val,trfm_dict_kwargs = get_trfm_dict_kwargs(col_names,pipelineOnly,
                                                                                                      trfm_dict_kwargs,model,model_name)
        if 'keras' in str(model):

            KModelObj=toExportDict[model_name]

            if 'model_graph' in KModelObj:
                model_graph = KModelObj['model_graph']
                with model_graph.as_default():
                    tf_session = KModelObj['tf_session']
                    with tf_session.as_default():
                        KerasPMML = KerasToPmml(model.model,model_name=pmml_f_name,targetVarName=target_name)
                            
            else:
                KerasPMML = KerasToPmml(model,model_name=pmml_f_name,targetVarName=target_name)

            model_obj = KerasPMML.DeepNetwork[0]
            model_obj.modelName = model_name
            model_obj.taskType=tasktype
            models_dict['DeepNetwork'].append(model_obj)


        else:    
            #model = pipeline.steps[-1][1]
            #ppln_sans_predictor = pipeline.steps[:-1]
            #derived_col_names,categoric_values,mining_imp_val,trfm_dict_kwargs = get_trfm_dict_kwargs(col_names,pipelineOnly,
            #                                                                                          trfm_dict_kwargs,modelobj,model_name)

            if ('XGBRegressor' in str(model)) or ('XGBClassifier' in str(model)):
                PMML_kwargs = xgboost_to_pmml(model,
                                            derived_col_names,
                                            col_names,
                                            target_name,
                                            mining_imp_val,
                                            categoric_values,
                                            tasktype)
            
            elif ('LGBMRegressor' in str(model)) or ('LGBMClassifier' in str(model)):
                PMML_kwargs = lgb_to_pmml(model,
                                            derived_col_names,
                                            col_names,
                                            target_name,
                                            mining_imp_val,
                                            categoric_values,
                                            tasktype)                                
            
            elif ('Booster' in str(model)):
                PMML_kwargs = ext(model,tasktype,target_name)
            
            else:
                PMML_kwargs = get_PMML_kwargs(model,
                                                derived_col_names,
                                                col_names,
                                                target_name,
                                                mining_imp_val,
                                                categoric_values,
                                                tasktype)
            model_obj = list(PMML_kwargs.values())[0][0]
            model_obj.modelName = model_name
            key = list(PMML_kwargs.keys())[0]
            if key in models_dict:
                models_dict[key].append(model_obj)
            else:
                PMML_kwargs = {key:[model_obj]}
                models_dict.update(PMML_kwargs)

            data_dicts.append(get_data_dictionary(model, col_names, target_name, categoric_values))
                   
    
    pmml = pml.PMML(
        version=get_version(),
        Header=get_header(),
        MiningBuildTask=get_mining_buildtask(toExportDict),
        DataDictionary=get_data_dictionary_values(data_dicts),
        script = get_script_execution(toExportDict),
        **trfm_dict_kwargs,
        **models_dict
    )
    pmml.export(outfile=open(pmml_f_name, "w"), level=0)


def get_trfm_dict_kwargs(col_names,pipelineOnly,trfm_dict_kwargs,model,model_name):
    if isinstance(col_names, np.ndarray):
        col_names = col_names.tolist()
    #ppln_sans_predictor = pipeline.steps[:-1]
    ppln_sans_predictor = pipelineOnly.steps
    derived_col_names = col_names
    categoric_values = tuple()
    mining_imp_val = tuple()
    if ppln_sans_predictor:
        pml_pp = pp.get_preprocess_val(ppln_sans_predictor, col_names, model, model_name)
        trfm_dict_kwargs['TransformationDictionary'].append(pml_pp['trfm_dict'])
        derived_col_names = pml_pp['derived_col_names']
        col_names = pml_pp['preprocessed_col_names']
        categoric_values = pml_pp['categorical_feat_values']
        mining_imp_val = pml_pp['mining_imp_values']

    return derived_col_names,categoric_values,mining_imp_val,trfm_dict_kwargs

def processScript(scr):

    scr=scr.replace('&','&amp;')
    return scr

def get_data_dictionary_values(data_dicts):
    data_dicts = [x for x in data_dicts if x is not None]
    lst = []
    lislen = len(data_dicts)
    if lislen != 0:
        for indfile in data_dicts[0].DataField:
            lst.append(indfile.get_name())
    if lislen == 0:
        datadict = None
    elif lislen == 1:
        datadict = data_dicts[0]
    else:
        for dd in range(1,lislen):
            for indfile in data_dicts[dd].DataField:
                if indfile.get_name() in lst and len(indfile.get_Value())==0:
                    pass
                else:
                    data_dicts[0].add_DataField(indfile)
                    lst.append(indfile.get_name())
        datadict = data_dicts[0]
    return datadict

def get_script_execution(toExportDict):

    # Script execution
    scrps = []
    for model_name in toExportDict.keys():
        if toExportDict[model_name]['preProcessingScript'] is not None:
            lstlen = len(toExportDict[model_name]['preProcessingScript']['scripts'])
            for leng in range(lstlen):
                scrps.append(pml.script(content=processScript(toExportDict[model_name]['preProcessingScript']['scripts'][leng]), 
                                        for_= model_name, 
                                        class_ = 'preprocessing',
                                        scriptPurpose = toExportDict[model_name]['preProcessingScript']['scriptpurpose'][leng]
                                        ))
        if toExportDict[model_name]['postProcessingScript'] is not None:
            lstlen = len(toExportDict[model_name]['postProcessingScript']['scripts'])
            for leng in range(0,lstlen):
                scrps.append(pml.script(content=processScript(toExportDict[model_name]['postProcessingScript']['scripts'][leng]), 
                                        for_= model_name, 
                                        class_ = 'postprocessing',
                                        scriptPurpose = toExportDict[model_name]['postProcessingScript']['scriptpurpose'][leng]
                                    ))

    return scrps

def get_entire_string(pipe0):
    pipe_steps = pipe0.steps
    pipe_memory = 'memory=' + str(pipe0.memory)
    df_container = ''
    pipe_container = ''
    for step_idx, step in enumerate(pipe_steps):
        pipe_step_container = ''
        step_name = step[0]
        step_item = step[1]
        if step_item.__class__.__name__ == "DataFrameMapper":
            df_default_val = "default=" + str(step_item.default)
            df_out_val = "df_out=" + str(step_item.df_out)
            input_df_val = "input_df=" + str(step_item.input_df)
            sparse_val = "sparse=" + str(step_item.sparse)
            for feature in step_item.features:
                if not df_container:
                    df_container = df_container + str(feature)
                else:
                    df_container = df_container + ',' + str(feature)
            df_container = '[' + df_container + ']'
            df_container = 'features=' + df_container
            df_container = df_default_val + ',' + df_out_val + ',\n\t' + df_container
            df_container = df_container + ',\n\t' + input_df_val + ',' + sparse_val
            df_container = '(' + df_container + ')'
            df_container = 'DataFrameMapper' + df_container
            df_container = '\'' + step_name + '\'' + ',' + df_container
            df_container = '(' + df_container + ')'
        else:
            pipe_step_container = '\'' + step_name + '\'' + ',' + str(step_item)
            pipe_step_container = '(' + pipe_step_container + ')'
            if not pipe_container:
                pipe_container = pipe_container + pipe_step_container
            else:
                pipe_container = pipe_container + ',' + pipe_step_container
    if df_container:
        pipe_container = df_container + ',' + pipe_container
    pipe_container = '[' + pipe_container + ']'
    pipe_container = 'steps=' + pipe_container
    pipe_container = pipe_memory + ',\n    ' + pipe_container
    pipe_container = 'Pipeline(' + pipe_container + ')'

    return pipe_container

def get_mining_buildtask(toExportDict):
    extension = []
    for model_name in toExportDict.keys():
        pipeline = toExportDict[model_name]['pipelineObj']
        if 'keras' in str(pipeline):
            pass
        else:
            if pipeline:
                pipeline = get_entire_string(pipeline)
                extension.append(pml.Extension(value=pipeline,for_=model_name,name="preprocessingPipeline"))
        modelobj = toExportDict[model_name]['modelObj']
        modelobj = str(modelobj)
        extension.append(pml.Extension(value=modelobj,for_=model_name,name="modelObject"))
        if toExportDict[model_name]['hyperparameters']:
            extension.append(pml.Extension(value=toExportDict[model_name]['hyperparameters'],for_=model_name,name="hyperparameters"))
    mining_bld_task = pml.MiningBuildTask(Extension = extension)
    return mining_bld_task


def any_in(seq_a, seq_b):
    return any(elem in seq_b for elem in seq_a)


def get_PMML_kwargs(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, tasktype):

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

    Returns
    -------
    algo_kwargs : Dictionary
        Get the PMML model argument based on scikit learn model object
    """
    skl_mdl_super_cls_names = get_super_cls_names(model)
    # regression_model_names = ('LinearRegression','LinearSVR')
    # regression_mining_model_names = ('LogisticRegression', 'RidgeClassifier','LinearDiscriminantAnalysis', \
    #                                     'SGDClassifier','LinearSVC',)
    regression_model_names = ('LinearRegression', 'LogisticRegression', 'RidgeClassifier', 'SGDClassifier',
                              'LinearDiscriminantAnalysis','LinearSVC','LinearSVR')
    tree_model_names = ('BaseDecisionTree',)
    support_vector_model_names = ('SVC', 'SVR')
    anomaly_model_names = ('OneClassSVM',)
    naive_bayes_model_names = ('GaussianNB',)
    mining_model_names = ('RandomForestRegressor', 'RandomForestClassifier', 'GradientBoostingClassifier',
                            'GradientBoostingRegressor','IsolationForest')
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
                                                    tasktype)}
    # elif any_in(regression_mining_model_names, skl_mdl_super_cls_names):
    #     if len(model.classes_) == 2:
    #         algo_kwargs = {'RegressionModel': get_regrs_models(model,
    #                                                        derived_col_names,
    #                                                        col_names,
    #                                                        target_name,
    #                                                        mining_imp_val,
    #                                                        categoric_values,
    #                                                        tasktype)}
    #     else:
    #         algo_kwargs = {'MiningModel': get_reg_mining_models(model,
    #                                                             derived_col_names,
    #                                                             col_names,
    #                                                             target_name,
    #                                                             mining_imp_val,
    #                                                             categoric_values,
    #                                                             tasktype)}
    elif any_in(regression_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'RegressionModel': get_regrs_models(model,
                                                           derived_col_names,
                                                           col_names,
                                                           target_name,
                                                           mining_imp_val,
                                                           categoric_values,
                                                           tasktype)}
    elif any_in(support_vector_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'SupportVectorMachineModel':
                           get_supportVectorMachine_models(model,
                                                           derived_col_names,
                                                           col_names,
                                                           target_name,
                                                           mining_imp_val,
                                                           categoric_values,
                                                           tasktype)}
    elif any_in(mining_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'MiningModel': get_ensemble_models(model,
                                                          derived_col_names,
                                                          col_names,
                                                          target_name,
                                                          mining_imp_val,
                                                          categoric_values,
                                                          tasktype)}
    elif any_in(neurl_netwk_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'NeuralNetwork': get_neural_models(model,
                                                          derived_col_names,
                                                          col_names,
                                                          target_name,
                                                          mining_imp_val,
                                                          categoric_values,
                                                          tasktype)}
    elif any_in(naive_bayes_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'NaiveBayesModel': get_naiveBayesModel(model,
                                                              derived_col_names,
                                                              col_names,
                                                              target_name,
                                                              mining_imp_val,
                                                              categoric_values,
                                                              tasktype)}
    elif any_in(nearest_neighbour_names, skl_mdl_super_cls_names):
        algo_kwargs = {'NearestNeighborModel':
                           get_nearestNeighbour_model(model,
                                                      derived_col_names,
                                                      col_names,
                                                      target_name,
                                                      mining_imp_val,
                                                      categoric_values,
                                                      tasktype)}
    elif any_in(anomaly_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'AnomalyDetectionModel':
                            get_anomalydetection_model(model,
                                                        derived_col_names,
                                                        col_names,
                                                        target_name,
                                                        mining_imp_val,
                                                        categoric_values,
                                                        tasktype)}
    elif any_in(clustering_model_names, skl_mdl_super_cls_names):
        algo_kwargs = {'ClusteringModel':
                            get_clustering_model(model,
                                                    derived_col_names,
                                                    col_names,
                                                    target_name,
                                                    mining_imp_val,
                                                    categoric_values,
                                                    tasktype
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

    Returns
    -------
    model_kwargs : Dictionary
        Returns  function name, MiningSchema and Output of the sk_model object
    """
    model_kwargs = dict()
    model_kwargs['functionName'] = get_mining_func(model)
    model_kwargs['MiningSchema'] = get_mining_schema(model, col_names, target_name, mining_imp_val, categoric_values)
    if model.__class__.__name__ == 'IsolationForest':
        model_kwargs['Output']=get_anomaly_detection_output(model)
    else:
        model_kwargs['Output'] = get_output(model, target_name)

    return model_kwargs


def get_reg_mining_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, tasktype):
    num_classes = len(model.classes_)
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values)

    mining_model = pml.MiningModel(modelName=model.__class__.__name__, taskType=tasktype,**model_kwargs)
    inner_mining_schema = [mfield for mfield in model_kwargs['MiningSchema'].MiningField if mfield.usageType != 'target']
    segmentation = pml.Segmentation(multipleModelMethod="modelChain")
    for idx in range(num_classes):
        segment = pml.Segment(id=str(idx+1),True_=pml.True_())
        segment.RegressionModel = pml.RegressionModel(
            functionName='regression',
            MiningSchema=pml.MiningSchema(
                MiningField=inner_mining_schema
                ),
            Output=pml.Output(
                OutputField=[
                    pml.OutputField(
                        name="probablity_"+str(idx),
                        optype="continuous",
                        dataType="double"
                        )
                    ]
                ),
            RegressionTable=get_reg_tab_for_reg_mining_model(model,derived_col_names,idx)
        )
        if model.__class__.__name__ != 'LinearSVC':
            segment.RegressionModel.normalizationMethod = "logit"
        segmentation.add_Segment(segment)

    last_segment = pml.Segment(id=str(num_classes+1),True_=pml.True_())
    mining_flds_for_last = [pml.MiningField(name="probablity_"+str(idx)) for idx in range(num_classes)]
    mining_flds_for_last.append(pml.MiningField(name=target_name,usageType="target"))
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
        functionName="classification",
        MiningSchema=mining_schema_for_last,
        RegressionTable=reg_tab_for_last
    )
    if model.__class__.__name__ != 'LinearSVC':
        last_segment.RegressionModel.normalizationMethod = "simplemax"
    segmentation.add_Segment(last_segment)
    mining_model.set_Segmentation(segmentation)
    return [mining_model]


def get_reg_tab_for_reg_mining_model(model, col_names, index):
    reg_tab = pml.RegressionTable(intercept="{:.16f}".format(model.intercept_[index]))
    for idx, coef in enumerate(model.coef_[index]):
        reg_tab.add_NumericPredictor(pml.NumericPredictor(name=col_names[idx],coefficient="{:.16f}".format(coef)))
    return [reg_tab]


def get_anomalydetection_model(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, tasktype):
    """
    It returns the KMean Clustering model element.

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


    Returns
    -------
    anomaly_detection_model :List
        Returns an anomaly detection model within a list

    """
    anomaly_detection_model = list()
    if 'OneClassSVM' in str(model.__class__):
        anomaly_detection_model.append(
            pml.AnomalyDetectionModel(
                modelName=model.__class__.__name__,
                algorithmType="ocsvm",
                functionName="regression",
                MiningSchema=get_mining_schema(model, col_names, target_name, mining_imp_val,categoric_values),
                Output=get_anomaly_detection_output(model),
                taskType=tasktype,
                SupportVectorMachineModel=get_supportVectorMachine_models(model,
                                                            derived_col_names,
                                                            col_names,
                                                            target_name,
                                                            mining_imp_val,
                                                            categoric_values,tasktype)[0]
            )
        )
    # else:
    #     anomaly_detection_model.append(
    #         pml.AnomalyDetectionModel(
    #             modelName="IsolationForests",
    #             algorithmType="iforest",
    #             functionName="regression",
    #             MiningSchema=get_mining_schema(model, col_names, target_name, mining_imp_val),
    #             Output=get_anomaly_detection_output(model),
    #             ParameterList=pml.ParameterList(Parameter=[pml.Parameter(
    #                                         name="training_data_count",
    #                                         value=model.max_samples_)]),
    #             MiningModel=get_ensemble_models(model,
    #                                         derived_col_names,
    #                                         col_names,
    #                                         target_name,
    #                                         mining_imp_val,
    #                                         categoric_values)[0]
    #         )
    # )
    return anomaly_detection_model


def get_anomaly_detection_output(model):
    """

    Parameters
    ----------

    Returns
    -------
    output_fields :
        Returns  an Output instance of anomaly detection model
    """
    output_fields = list()

    if 'OneClassSVM' in str(model.__class__):
        output_fields.append(pml.OutputField(
            name="anomalyScore",
            feature="predictedValue",
            optype="continuous",
            dataType="float"))
        output_fields.append(pml.OutputField(
            name="anomaly",
            feature="anomaly",
            optype="categorical",
            dataType="boolean",
            threshold="0"
        ))

    else:
        n = model.max_samples_
        eulers_gamma = 0.577215664901532860606512090082402431

        output_fields.append(pml.OutputField(name="rawAnomalyScore", 
                                            optype="continuous", 
                                            dataType="double",
                                            feature="predictedValue",
                                            isFinalResult="false"))

        output_fields.append(pml.OutputField(name="normalizedAnomalyScore",
                                            optype="continuous",
                                            dataType="double",
                                            feature="transformedValue",
                                            isFinalResult="false", 
                                            Apply=pml.Apply(function="/", 
                                                            FieldRef=[pml.FieldRef(field="rawAnomalyScore")], 
                                                            Constant=[pml.Constant(dataType="double",
                                                                                   valueOf_=(2.0*(math.log(n-1.0)+eulers_gamma))-
                                                                                            (2.0*((n-1.0)/n)))])))

        appl_inner_inner = pml.Apply(function="*")
        cnst = pml.Constant(dataType="double", valueOf_=-1.0)
        fldref = pml.FieldRef(field="normalizedAnomalyScore")
        cnst.original_tagname_ = 'Constant'
        appl_inner_inner.add_FieldRef(cnst)
        appl_inner_inner.add_FieldRef(fldref)

        appl_inner = pml.Apply(function='pow')
        cnst = pml.Constant(dataType="double", valueOf_=2.0)
        cnst.original_tagname_ = 'Constant'
        appl_inner.add_FieldRef(cnst)
        appl_inner_inner.original_tagname_='Apply'
        appl_inner.add_FieldRef(appl_inner_inner)

        appl_outer = pml.Apply(function="-")
        cnst = pml.Constant(dataType="double", valueOf_=0.5)
        cnst.original_tagname_ = 'Constant'
        appl_outer.add_FieldRef(cnst)
        appl_inner.original_tagname_='Apply'
        appl_outer.add_FieldRef(appl_inner)

        output_fields.append(pml.OutputField(name="decisionFunction",
                                            optype="continuous",
                                            dataType="double",
                                            feature="transformedValue",
                                            isFinalResult="false", 
                                            Apply=appl_outer))

        output_fields.append(pml.OutputField(name="outlier",
                                            optype="categorical",
                                            dataType="boolean",
                                            feature="transformedValue",
                                            isFinalResult="true", 
                                            Apply=pml.Apply(function="greaterThan", 
                                                            FieldRef=[pml.FieldRef(field="decisionFunction")],
                                                            Constant=[pml.Constant(dataType="double", 
                                                                                    valueOf_="{:.16f}".format(model.threshold_))])))

    return pml.Output(OutputField=output_fields)


def get_clustering_model(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype):
    """
    It returns the KMean Clustering model element.

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


    Returns
    -------
    clustering_models :List
        Returns a KMean Clustering model within a list

    """

    clustering_models = list()
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    values, counts = np.unique(model.labels_,return_counts=True)
    model_kwargs["Output"] = get_output_for_clustering(values)
    clustering_models.append(
        pml.ClusteringModel(
            modelClass="centerBased",
            modelName=model.__class__.__name__,
            numberOfClusters=get_cluster_num(model),
            ComparisonMeasure=get_comp_measure(),
            ClusteringField=get_clustering_flds(derived_col_names),
            Cluster=get_cluster_vals(model,counts),
            taskType=tasktype,
            **model_kwargs

        )
    )

    return clustering_models


def get_output_for_clustering(values):
    """

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------
    output_fields :List
        Returns a list of OutputField
    """
    output_fields = list()
    output_fields.append(pml.OutputField(name="cluster", optype="categorical",dataType="string",feature="predictedValue"))
    for idx, val in enumerate(values):
        output_fields.append(
            pml.OutputField(
                name="affinity("+str(idx)+")",
                optype="continuous",
                dataType="double",
                feature="entityAffinity",
                value=str(val)
            )
        )
    return pml.Output(OutputField=output_fields)
        


def get_cluster_vals(model,counts):
    """

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------
    cluster_flds :List
        Returns a list of Cluster instances

    """
    centroids = model.cluster_centers_
    cluster_flds = []
    for centroid_idx in range(centroids.shape[0]):
        centroid_values = ""
        centroid_flds = pml.ArrayType(type_="real")
        for centroid_cordinate_idx in range(centroids.shape[1]):
            centroid_flds.content_[0].value = centroid_values + "{:.16f}".format(centroids[centroid_idx][centroid_cordinate_idx])
            centroid_values = centroid_flds.content_[0].value + " "
        cluster_flds.append(pml.Cluster(id=str(centroid_idx), Array=centroid_flds,size=str(counts[centroid_idx])))
    return cluster_flds


def get_cluster_num(model):
    """

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

    Parameters
    ----------

    Returns
    -------

        Returns an instance of comparision measure

    """
    comp_equation = pml.euclidean()
    return pml.ComparisonMeasure(euclidean=comp_equation, kind="distance")


def get_clustering_flds(col_names):
    """

    Parameters
    ----------
    col_names :
        Contains list of feature/column names.

    Returns
    -------

    clustering_flds: List

        Returns the list containing clustering field instances

    """
    clustering_flds = []
    for name in col_names:
        clustering_flds.append(pml.ClusteringField(field=str(name)))
    return clustering_flds


def get_nearestNeighbour_model(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype):
    
    """
    It returns the Nearest Neighbour model element.

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
    

    Returns
    -------
    nearest_neighbour_model :
        Returns a nearest neighbour model instance
        
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    nearest_neighbour_model = list()
    nearest_neighbour_model.append(
        pml.NearestNeighborModel(
            modelName=model.__class__.__name__,
            continuousScoringMethod='average',
            algorithmName="KNN",
            numberOfNeighbors=model.n_neighbors,
            KNNInputs=get_knn_inputs(derived_col_names),
            ComparisonMeasure=get_comparison_measure(model),
            TrainingInstances=get_training_instances(model, derived_col_names, target_name),
            taskType=tasktype,
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
        Returns a TrainingInstances instance
        
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
        Returns a InlineTable instance.
        
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
        Returns a InstanceFields instance
        
    """
    instance_fields = list()
    instance_fields.append(pml.InstanceField(field=target_name, column="y"))
    for (index, name) in enumerate(derived_col_names):
        instance_fields.append(pml.InstanceField(field=str(name), column="x" + str(index + 1)))
    return pml.InstanceFields(InstanceField=instance_fields)


def get_comparison_measure(model):


    """
    It return the Comparison measure element.

    Parameters
    ----------
    model :
        An instance of Scikit-learn model.

    Returns
    -------
    comp_measure :
        Returns a ComparisonMeasure instance.
        
    """
    if model.effective_metric_ == 'euclidean':
        comp_measure = pml.ComparisonMeasure(euclidean=pml.euclidean(), kind="distance")
    elif model.effective_metric_ == 'minkowski':
        comp_measure = pml.ComparisonMeasure(minkowski=pml.minkowski(p_parameter=model.p), kind="distance")
    elif model.effective_metric_ in ['manhattan','cityblock']:
        comp_measure = pml.ComparisonMeasure(cityBlock=pml.cityBlock(), kind="distance")
    elif model.effective_metric_ == 'sqeuclidean':
        comp_measure = pml.ComparisonMeasure(squaredEuclidean=pml.squaredEuclidean(), kind="distance")
    elif model.effective_metric_ == 'chebyshev':
        comp_measure = pml.ComparisonMeasure(chebychev=pml.chebychev(), kind="distance")
    elif model.effective_metric_ == 'matching':
        comp_measure = pml.ComparisonMeasure(simpleMatching=pml.simpleMatching(), kind="similarity")
    elif model.effective_metric_ == 'jaccard':
        comp_measure = pml.ComparisonMeasure(jaccard=pml.jaccard(), kind="similarity")
    elif model.effective_metric_ == 'rogerstanimoto':
        comp_measure = pml.ComparisonMeasure(tanimoto=pml.tanimoto(), kind="similarity")
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
        Returns a KNNInputs instance.
        
    """
    knnInput = list()
    for name in col_names:
        knnInput.append(pml.KNNInput(field=str(name)))
    return pml.KNNInputs(KNNInput=knnInput)


def get_naiveBayesModel(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype):

    """
    It returns the Naive Bayes Model element of the model.

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

    Returns
    -------
    naive_bayes_model : List
        Returns the NaiveBayesModel
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    naive_bayes_model = list()
    naive_bayes_model.append(pml.NaiveBayesModel(
        modelName=model.__class__.__name__,
        BayesInputs=get_bayes_inputs(model, derived_col_names),
        BayesOutput=get_bayes_output(model, target_name),
        threshold=get_threshold(),
        taskType=tasktype,
        **model_kwargs
    ))
    return naive_bayes_model


def get_threshold():
    """
    It returns the Threshold value.

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
        Returns a BayesOutput instance
        
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
    It returns the Bayes Input element of the model .
    
    Parameters
    ----------
    model :
        An instance of Scikit-learn model.
    derived_col_names : List
        Contains column names after preprocessing.

    Returns
    -------
    bayes_inputs :
        Returns a BayesInput instance.

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
 									mining_imp_val, categoric_values, tasktype):
    
    """
    It returns the Support Vector Machine Model element.
    
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

    Returns
    -------
    supportVector_models : List
        Returns SupportVectorMachineModel elements which contains classificationMethod,
        VectorDictionary, SupportVectorMachine, kernelType
        
    """
    model_kwargs = get_model_kwargs(model, col_names, target_names, mining_imp_val,categoric_values)
    supportVector_models = list()
    kernel_type = get_kernel_type(model)
    supportVector_models.append(pml.SupportVectorMachineModel(
        modelName=model.__class__.__name__,
        classificationMethod=get_classificationMethod(model),
        VectorDictionary=get_vectorDictionary(model, derived_col_names, categoric_values),
        SupportVectorMachine=get_supportVectorMachine(model),
        taskType=tasktype,
        **kernel_type,
        **model_kwargs
    ))
    # supportVector_models[0].export(sys.stdout,0," ")

    return supportVector_models

def get_model_name(model):
    if 'OneClassSVM' in str(model.__class__):
        return 'ocsvm'
    elif 'IsolationForest' in str(model.__class__):
        return 'iforest'
    elif 'XGB' in str(model.__class__):
        return 'XGBoostModel'
    elif 'LGB' in str(model.__class__):
        return 'LightGBModel'
    elif 'GradientBoosting' in str(model.__class__):
        return 'GradientBoostingModel'
    elif 'RandomForest' in str(model.__class__):
        return 'RandomForestModel'

def get_ensemble_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, tasktype):
    
    """
    It returns the Mining Model element of the model

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

    Returns
    -------
    mining_models : List
        Returns the MiningModel of the respective ensemble model
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    if model.__class__.__name__ == 'GradientBoostingRegressor':
        model_kwargs['Targets'] = get_targets(model, target_name)

    mining_fields = model_kwargs['MiningSchema'].MiningField
    new_mining_fields = list()
    if model.__class__.__name__ != 'IsolationForest':
        for idx, imp_ in enumerate(model.feature_importances_):
            if imp_ > 0:
                new_mining_fields.append(mining_fields[idx])
    else:
        for idx in range(len(col_names)):
            new_mining_fields.append(mining_fields[idx])
    for fld in mining_fields:
        if fld.usageType == 'target':
            new_mining_fields.append(fld)
    model_kwargs['MiningSchema'].MiningField = new_mining_fields

        
    mining_models = list()
    mining_models.append(pml.MiningModel(
        modelName=model.__class__.__name__,
        Segmentation=get_outer_segmentation(model, derived_col_names, col_names, target_name,
                                            mining_imp_val, categoric_values,tasktype),
        taskType=tasktype,
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
        Returns a Target instance.
    """
    if model.__class__.__name__ == 'GradientBoostingRegressor':
        targets = pml.Targets(
            Target=[
                pml.Target(
                    field=target_name,
                    rescaleConstant="{:.16f}".format(model.init_.mean),
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
    It returns the name of the Multiple Model Chain element of the model.

    Parameters
    ----------
    model :
        A Scikit-learn model instance

    Returns
    -------
    The multiple model method for a mining model.
        
    """
    if model.__class__.__name__ == 'GradientBoostingClassifier':
        return 'modelChain'
    elif model.__class__.__name__ == 'GradientBoostingRegressor':
        return 'sum'
    elif model.__class__.__name__ == 'RandomForestClassifier':
        return 'majorityVote'
    elif model.__class__.__name__ in ['RandomForestRegressor','IsolationForest']:
        return 'average'


def get_outer_segmentation(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,tasktype):
    
    """
    It returns the Segmentation element of the model.

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
        A segmentation instance.
        
    """
    segmentation = pml.Segmentation(
        multipleModelMethod=get_multiple_model_method(model),
        Segment=get_segments(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,tasktype)
    )
    return segmentation


def get_segments(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,tasktype):

    """
    It returns the Segment element of the model.

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
        A list of segment instances.
        
    """
    segments = None
    if 'GradientBoostingClassifier' in str(model.__class__):
        segments = get_segments_for_gbc(model, derived_col_names, col_names, target_name,
                                        mining_imp_val, categoric_values,tasktype)
    else:
        segments = get_inner_segments(model, derived_col_names, col_names, 0)
    return segments


def get_segments_for_gbc(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,tasktype):
    
    """
    It returns list of Segments element of the model.

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
        Get the Segments for the Segmentation element.
        
    """
    segments = list()
    out_field_names = list()
    for estm_idx in range(len(model.estimators_[0])):
        mining_fields_for_first = list()
        # for name in col_names:
        for idx,imp_ in enumerate(model.feature_importances_):
            # mining_fields_for_first.append(pml.MiningField(name=name))
            if imp_ > 0:
                mining_fields_for_first.append(pml.MiningField(name=col_names[idx]))

        miningschema_for_first = pml.MiningSchema(MiningField=mining_fields_for_first)
        output_fields = list()
        output_fields.append(
            pml.OutputField(
                name='decisionFunction(' + str(estm_idx) + ')',
                feature='predictedValue',
                dataType="double",
                isFinalResult=False
            )
        )
        if len(model.classes_) == 2:
            output_fields.append(
                pml.OutputField(
                    name='transformedDecisionFunction(0)',
                    feature='transformedValue',
                    dataType="double",
                    isFinalResult=True,
                    Apply=pml.Apply(
                        function="+",
                        Constant=[pml.Constant(
                            dataType="double",
                            valueOf_="{:.16f}".format(model.init_.prior)
                        )],
                        Apply_member=[pml.Apply(
                            function="*",
                            Constant=[pml.Constant(
                                dataType="double",
                                valueOf_="{:.16f}".format(model.learning_rate)
                            )],
                            FieldRef=[pml.FieldRef(
                                field="decisionFunction(0)",
                            )]
                        )]
                    )
                )
            )
        else:
            output_fields.append(
                pml.OutputField(
                    name='transformedDecisionFunction(' + str(estm_idx) + ')',
                    feature='transformedValue',
                    dataType="double",
                    isFinalResult=True,
                    Apply=pml.Apply(
                        function="+",
                        Constant=[pml.Constant(
                            dataType="double",
                            valueOf_="{:.16f}".format(model.init_.priors[estm_idx])
                        )],
                        Apply_member=[pml.Apply(
                            function="*",
                            Constant=[pml.Constant(
                                dataType="double",
                                valueOf_="{:.16f}".format(model.learning_rate)
                            )],
                            FieldRef=[pml.FieldRef(
                                field="decisionFunction(" + str(estm_idx) + ")",
                            )]
                        )]
                    )
                )
            )

        out_field_names.append('transformedDecisionFunction(' + str(estm_idx) + ')')
        segments.append(
            pml.Segment(
                True_=pml.True_(),
                id=str(estm_idx),
                MiningModel=pml.MiningModel(
                    functionName='regression',
                    modelName="MiningModel",
                    MiningSchema=miningschema_for_first,
                    Output=pml.Output(OutputField=output_fields),
                    Segmentation=pml.Segmentation(
                        multipleModelMethod="sum",
                        Segment=get_inner_segments(model, derived_col_names,
                                                   col_names, estm_idx)
                    )
                )
            )
        )
    reg_model = get_regrs_models(model, out_field_names,out_field_names, target_name, mining_imp_val, categoric_values,tasktype)[0]
    reg_model.Output = None
    if len(model.classes_) == 2:
        reg_model.normalizationMethod="logit"
    else:
        reg_model.normalizationMethod="softmax"
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
    It returns the Inner segments of the model.
    
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
        Get the Segments for the Segmentation element.
        
    """
    segments = list()
    for estm_idx in range(model.n_estimators):
        if np.asanyarray(model.estimators_).ndim == 1:
            estm = model.estimators_[estm_idx]
        else:
            estm = model.estimators_[estm_idx][index]
        tree_features = estm.tree_.feature
        features_ = list()
        for feat in tree_features:
            if feat != -2 and feat not in features_:
                features_.append(feat)
        if len(features_) != 0:
            mining_fields = list()
            # for feat in col_names:
            feature_importances = estm.tree_.compute_feature_importances()
            for idx,imp_ in enumerate(feature_importances):
                if imp_ > 0:
                # mining_fields.append(pml.MiningField(name=feat))
                    mining_fields.append(pml.MiningField(name=col_names[idx]))
            segments.append(
                pml.Segment(
                    True_=pml.True_(),
                    id=str(estm_idx),
                    TreeModel=pml.TreeModel(
                        modelName=estm.__class__.__name__,
                        functionName=get_mining_func(estm),
                        splitCharacteristic="multiSplit",
                        MiningSchema=pml.MiningSchema(MiningField = mining_fields),
                        Node=get_node(estm, derived_col_names, model)
                    )
                )
            )
    return segments


def get_classificationMethod(model):
    
    """
    It returns the Classification Model name of the model.
    
    Parameters
    ----------
    model :
        A Scikit-learn model instance.

    Returns
    -------
    Returns the classification method of the SVM model
        
    """
    if model.__class__.__name__ == 'SVC':
        return 'OneAgainstOne'
    else:
        return 'OneAgainstAll'


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
        A Vector Dictionary instance.
        
    """
    model_coef = model.C
    fieldref_element = get_vectorfields(model_coef, derived_col_names, categoric_values)
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




def get_vectorfields(model_coef, feat_names, categoric_values):

    """
     It return the Vector Fields .

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
     Returns the Vector Dictionary instance for Support Vector model.

     """
    der_fld_len = len(feat_names)
    der_fld_idx = 0
    row_idx = -1
    predictors = list()
    if categoric_values:
        class_lbls = categoric_values[0]
        class_attribute = categoric_values[1]
    while der_fld_idx < der_fld_len:
        if is_labelbinarizer(feat_names[der_fld_idx]):
            if not is_stdscaler(feat_names[der_fld_idx]):
                class_id = get_classid(class_attribute, feat_names[der_fld_idx])
                cat_predictors = get_categoric_pred(feat_names[der_fld_idx],row_idx, der_fld_idx, model_coef, class_lbls[class_id],
                                                    class_attribute[class_id])
                for predictor in cat_predictors:
                    predictors.append(predictor)

                if len(class_lbls[class_id]) == 2:
                    incrementor = 1
                else:
                    incrementor = len(class_lbls[class_id])
                der_fld_idx = der_fld_idx + incrementor
            else:
                vectorfields_element = pml.FieldRef(field=feat_names[der_fld_idx])
                predictors.append(vectorfields_element)
                der_fld_idx += 1

        elif is_onehotencoder(feat_names[der_fld_idx]):
            if not is_stdscaler(feat_names[der_fld_idx]):
                class_id = get_classid(class_attribute, feat_names[der_fld_idx])
                cat_predictors = get_categoric_pred(feat_names[der_fld_idx],row_idx, der_fld_idx, model_coef, class_lbls[class_id],
                                                    class_attribute[class_id])
                for predictor in cat_predictors:
                    predictors.append(predictor)

                incrementor = len(class_lbls[class_id])
                der_fld_idx = der_fld_idx + incrementor
            else:
                vectorfields_element = pml.FieldRef(field=feat_names[der_fld_idx])
                predictors.append(vectorfields_element)
                der_fld_idx += 1

        else:
            vectorfields_element = pml.FieldRef(field=feat_names[der_fld_idx])
            predictors.append(vectorfields_element)
            der_fld_idx += 1

    return predictors

def is_onehotencoder(feat_name):
    """

    Parameters
    ----------
    feat_name : string
        Contains the name of the attribute

    Returns
    -------
        Returns a boolean value that states whether OneHotEncoder has been applied or not

    """
    if "oneHotEncoder" in feat_name:
        return True
    else:
        return False


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
    It return the Support Vector Machine element.
    
    Parameters
    ----------
    model :
        A Scikit-learn model instance.

    Returns
    -------
    support_vector_machines : List
        Get the Support Vector Machine element which conatains targetCategory,
        alternateTargetCategory, SupportVectors, Coefficients

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


def get_tree_models(model, derived_col_names, col_names, target_name, mining_imp_val,categoric_values,tasktype):

    """
    It return Tree Model element of the model

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

    Returns
    -------
    tree_models : List
        Get the TreeModel element.
        
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    tree_models = list()
    tree_models.append(pml.TreeModel(
        modelName=model.__class__.__name__,
        Node=get_node(model, derived_col_names),
        taskType=tasktype,
        **model_kwargs
    ))
    return tree_models


def get_neural_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values,tasktype):

    """
    It returns Neural Network element of the model.

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

    Returns
    -------
    neural_model : List
        Model attributes for PMML file.
        
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val,categoric_values)
    neural_model = list()
    neural_model.append(pml.NeuralNetwork(
        modelName=model.__class__.__name__,
        threshold='0',
        altitude='1.0',
        activationFunction=get_funct(model),
        NeuralInputs = get_neuron_input(derived_col_names),
        NeuralLayer = get_neural_layer(model, derived_col_names, target_name)[0],
        NeuralOutputs = get_neural_layer(model, derived_col_names, target_name)[1],
        **model_kwargs
    ))
    return neural_model


def get_funct(sk_model):

    """
    It returns the activation fucntion of the model.
    
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
        a_fn = 'rectifier'
    return a_fn


def get_regrs_models(model, derived_col_names, col_names, target_name, mining_imp_val, categoric_values, tasktype):

    """
    It returns the Regression Model element of the model
    
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
    regrs_models : List
        Returns a regression model of the respective model
    """
    model_kwargs = get_model_kwargs(model, col_names, target_name, mining_imp_val, categoric_values)
    if model.__class__.__name__ not in ['LinearRegression','LinearSVR']: 
        model_kwargs['normalizationMethod'] = 'logit'
    regrs_models = list()
    regrs_models.append(pml.RegressionModel(
        modelName=model.__class__.__name__,
        RegressionTable=get_regrs_tabl(model, derived_col_names, target_name, categoric_values),
        taskType=tasktype,
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
        Returns a list of Regression Table.
        
    """
    merge = list()
    if hasattr(model, 'intercept_'):
        func_name = get_mining_func(model)
        inter = model.intercept_
        model_coef = model.coef_
        merge = list()
        target_classes = target_name
        row_idx = 0
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

        if len(inter) == 1:
            regr_predictor = get_regr_predictors(model_coef, row_idx, feature_names, categoric_values)
            merge.append(
                pml.RegressionTable(
                    intercept="{:.16f}".format(inter.item()),
                    targetCategory=target_cat,
                    NumericPredictor=regr_predictor
                )
            )
            if func_name != 'regression':
                merge.append(
                    pml.RegressionTable(
                        intercept="0.0",
                        targetCategory=target_classes[0]
                    )
                )
        else:
            for tgname, tg_idx in zip(np.unique(target_classes), range(len(np.unique(target_classes)))):
                row_idx = tg_idx
                regr_predictors = get_regr_predictors(model_coef, row_idx, feature_names, categoric_values)
                merge.append(
                    pml.RegressionTable(
                        intercept="{:.16f}".format(inter[tg_idx]),
                        targetCategory=tgname,
                        NumericPredictor=regr_predictors
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
    _getNode :
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
            simplePredicate = pml.SimplePredicate(field=fieldName, operator="lessOrEqual",
                                                  value="{:.16f}".format(tree.threshold[idx]))
            left_child = _getNode(tree.children_left[idx],prnt, simplePredicate)
            simplePredicate = pml.SimplePredicate(field=fieldName, operator="greaterThan",
                                                  value="{:.16f}".format(tree.threshold[idx]))
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
        Get the Output element.
        
    """

    mining_func = get_mining_func(model)
    output_fields = list()
    if not has_target(model):
        output_fields.append(pml.OutputField(
                name='predicted',
                feature="predictedValue",
                optype="categorical",
                dataType="double"
            ))
    else:
        alt_target_name = 'predicted_' + target_name
        if mining_func == 'classification':
            for cls in model.classes_:
                output_fields.append(pml.OutputField(
                    name='probability_' + str(cls),
                    feature="probability",
                    optype="continuous",
                    dataType="double",
                    value=str(cls)
                ))
            output_fields.append(pml.OutputField(
                name=alt_target_name,
                feature="predictedValue",
                optype="categorical",
                dataType="string"))
        else:
            output_fields.append(pml.OutputField(
                name=alt_target_name,
                feature="predictedValue",
                optype="continuous",
                dataType="double"))
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
    if not hasattr(model, 'classes_'):
        if hasattr(model,'n_clusters'):
            func_name = 'clustering'
        else:
            func_name = 'regression'
    else:
        if isinstance(model.classes_, np.ndarray):
            func_name = 'classification'
        else:
            func_name = 'regression'

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

    Returns
    -------
    MiningSchema :
        Get the MiningSchema element
        
    """
    if mining_imp_val:
        mining_attributes = mining_imp_val[0]
        mining_strategy = mining_imp_val[1]
        mining_replacement_val = mining_imp_val[2]
    n_features = len(feature_names)
    features_pmml_optype = ['continuous'] * n_features
    features_pmml_utype = ['active'] * n_features
    target_pmml_utype = 'target'
    mining_func = get_mining_func(model)
    if mining_func == 'classification':
        target_pmml_optype = 'categorical'
    elif mining_func == 'regression':
        target_pmml_optype = 'continuous'
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
                usageType='active',
                optype='categorical'
            ))
            mining_name_stored.append(cls_attr)
    for feat_name, feat_idx in zip(feature_names, range(len(feature_names))):
        if feat_name not in mining_name_stored:
            mining_flds.append(pml.MiningField(name=str(feat_name),
                                               optype=features_pmml_optype[feat_idx],
                                               usageType=features_pmml_utype[feat_idx]))
    if has_target(model):
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
        Returns the NeuralInputs element
        
    """
    neural_input = list()
    for features in feature_names:
        field_ref = pml.FieldRef(field = str(features))
        derived_flds = pml.DerivedField(optype = "continuous", dataType = "double", FieldRef = field_ref)
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
        Return the list of NeuralLayer elelemt.

    neural_output_element :
        Return the NeuralOutput element instance
        
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
    if hidden_layers[-1]==1 and 'MLPClassifier' in str(model.__class__):
        bias1=[1.0,0.0]
        weight1=[-1.0,1.0]
        con = list()
        linear = ['linear/1']
        i_d = ['true', 'false']
        con.append(pml.Con(from_ = input_features[0], weight = 1.0))
        neuron.append(pml.Neuron(id = linear[0], bias = ('0.0'), Con = con))
        all_neuron_layer.append(pml.NeuralLayer(activationFunction = "logistic", Neuron = neuron))
        neuron = list()
        con = list()
        for num in range(2):
            con.append(pml.Con(from_ = linear[0], weight = format(weight1[num])))
            neuron.append(pml.Neuron(id = i_d[num], bias = format(bias1[num]), Con = con))
            con = list()
        all_neuron_layer.append(pml.NeuralLayer(activationFunction = "identity", Neuron = neuron))
    if 'MLPClassifier' in str(model.__class__):
        neural_output = list()
        for values, count in zip(model.classes_, range(len(model.classes_))):
            norm_discrete = pml.NormDiscrete(field = target_name, value = str(values))
            derived_flds = pml.DerivedField(optype = "categorical", dataType = 'double',
                                    NormDiscrete = norm_discrete)
            if len(input_features)==1:
                class_node = pml.NeuralOutput(outputNeuron = input_features, DerivedField = derived_flds)
            else:
                class_node = pml.NeuralOutput(outputNeuron = input_features[count],DerivedField = derived_flds)
            neural_output.append(class_node)
        neural_output_element = pml.NeuralOutputs(numberOfOutputs = None, Extension = None,
                                                  NeuralOutput = neural_output)
    if 'MLPRegressor' in str(model.__class__):
        neural_output = list()
        fieldRef = pml.FieldRef(field = target_name)
        derived_flds = pml.DerivedField(optype = "continuous", dataType = "double", FieldRef = fieldRef)
        class_node = pml.NeuralOutput(outputNeuron = input_features, DerivedField = derived_flds)
        neural_output.append(class_node)
    neural_output_element = pml.NeuralOutputs(numberOfOutputs = None, Extension = None, NeuralOutput = neural_output)
    return all_neuron_layer, neural_output_element


def get_super_cls_names(model_inst):
    """
    It returns the set of Super class of the model.

    Parameters:
    -------
    model_inst:
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


def get_version():
    """
    It returns the pmml version .

    Returns
    -------
    version : String
        Returns the version of the pmml.

    """

    version = '4.4'
    return version

def get_header():

    """
    It returns the Header element of the pmml.

     Returns
     -------
     header :
         Returns the header of the pmml.

     """
    copyryt = "Copyright (c) 2019 Software AG"
    description = "Default Description"
    timestamp = pml.Timestamp(datetime.now())
    application=pml.Application(name="Nyoka",version=metadata.__version__)
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
    data_type=str(type(feat_value))
    if 'float' in data_type:
        return 'float'
    if 'int' in data_type:
        return 'integer'
    if 'long' in data_type:
        return 'long'
    if 'complex' in data_type:
        return 'complex'
    if 'str' in data_type:
        return 'string'

def get_data_dictionary(model, feature_names, target_name, categoric_values=None):

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
        Return the dataDictionary instance
        
    """
    categoric_feature_name = list()
    if categoric_values:
        categoric_labels = categoric_values[0]
        categoric_feature_name = categoric_values[1]
    target_attr_values = []
    n_features = len(feature_names)
    features_pmml_optype = ['continuous'] * n_features
    features_pmml_dtype = ['double'] * n_features

    mining_func = get_mining_func(model)

    if mining_func == 'classification':
        target_pmml_optype = 'categorical'
        target_pmml_dtype = get_dtype(model.classes_[0])
        target_attr_values = model.classes_.tolist()
    elif mining_func == 'regression':
        target_pmml_optype = 'continuous'
        target_pmml_dtype = 'double'

    data_fields = list()
    if categoric_values:
        for class_list, attr_for_class in zip(categoric_labels, categoric_feature_name):
            category_flds = pml.DataField(name=str(attr_for_class), optype="categorical",
                                        dataType=get_dtype(class_list[0]) if class_list else 'string')
            if class_list:
                for values in class_list:
                    category_flds.add_Value(pml.Value(value=str(values)))
            data_fields.append(category_flds)
    attr_without_class_attr = [feat_name for feat_name in feature_names if feat_name not in categoric_feature_name]
    for feature_idx, feat_name in enumerate(attr_without_class_attr):
        data_fields.append(pml.DataField(name=str(feat_name),
                                        optype=features_pmml_optype[feature_idx],
                                        dataType=features_pmml_dtype[feature_idx]))
    if has_target(model):
        class_node = pml.DataField(name=str(target_name), optype=target_pmml_optype,
                                dataType=target_pmml_dtype)

        for class_value in target_attr_values:
            class_node.add_Value(pml.Value(value=str(class_value)))
        data_fields.append(class_node)
    data_dict = pml.DataDictionary(numberOfFields=len(data_fields), DataField=data_fields)
    return data_dict


def has_target(model):
    target_less_models = ['KMeans','OneClassSVM','IsolationForest', ]
    if model.__class__.__name__  in target_less_models:
        return False
    else:
        return True


def get_regr_predictors(model_coef, row_idx, feat_names, categoric_values):
    """

    Parameters
    ----------
    model_coef : array
        Contains the estimators coefficient values
    row_idx : int
        Contains an integer value to differentiate between linear and svm models
    feat_names : list
        Contains the list of feature/column names
    categoric_values : tuple
        Contains Categorical attribute names and its values

    Returns
    -------
    predictors : list
        Returns a list with instances of nyoka numeric/categorical predictor class

    """

    der_fld_len = len(feat_names)
    der_fld_idx = 0
    predictors = list()
    if categoric_values:
        class_lbls = categoric_values[0]
        class_attribute = categoric_values[1]
    while der_fld_idx < der_fld_len:

        if is_labelbinarizer(feat_names[der_fld_idx]):
            if not is_stdscaler(feat_names[der_fld_idx]):
                class_id = get_classid(class_attribute, feat_names[der_fld_idx])
                cat_predictors = get_categoric_pred(feat_names[der_fld_idx], row_idx, der_fld_idx, model_coef,
                                                    class_lbls[class_id], class_attribute[class_id])
                for predictor in cat_predictors:
                    predictors.append(predictor)
                if len(class_lbls[class_id]) == 2:
                    incrementor = 1
                else:
                    incrementor = len(class_lbls[class_id])
                der_fld_idx = der_fld_idx + incrementor

            else:
                num_predictors = get_numeric_pred(row_idx, der_fld_idx, model_coef, feat_names[der_fld_idx])
                predictors.append(num_predictors)
                der_fld_idx += 1
        elif is_onehotencoder(feat_names[der_fld_idx]):
            if not is_stdscaler(feat_names[der_fld_idx]):
                class_id = get_classid(class_attribute, feat_names[der_fld_idx])
                cat_predictors = get_categoric_pred(feat_names[der_fld_idx], row_idx, der_fld_idx, model_coef,
                                                    class_lbls[class_id], class_attribute[class_id])
                for predictor in cat_predictors:
                    predictors.append(predictor)

                incrementor = len(class_lbls[class_id])
                der_fld_idx = der_fld_idx + incrementor
            else:
                vectorfields_element = pml.FieldRef(field=feat_names[der_fld_idx])
                predictors.append(vectorfields_element)
                der_fld_idx += 1

        else:
            num_predictors = get_numeric_pred(row_idx, der_fld_idx, model_coef, feat_names[der_fld_idx])
            predictors.append(num_predictors)
            der_fld_idx += 1
    return predictors

def get_classid(class_attribute, feat_name):
    """

    Parameters
    ----------
    class_attribute:
        Contains the name of the attribute/column that contains categorical values

    feat_name : string
        Contains the name of the attribute/column

    Returns
    -------
    class_idx:int
        Returns an integer value that will represent each categorical value

    """
    for class_idx,class_attr in enumerate(class_attribute):
        if class_attr in feat_name:
            return class_idx



def is_labelbinarizer(feat_name):
    """

    Parameters
    ----------
    feat_name : string
        Contains the name of the attribute

    Returns
    -------
        Returns a boolean value that states whether label binarizer has been applied or not

    """
    if "labelBinarizer" in feat_name or "one_hot_encoder" in feat_name:
        return True
    else:
        return False


def is_stdscaler(feat_name):
    """

    Parameters
    ----------
    feat_name : string
        Contains the name of the attribute

    Returns
    -------
        Returns a boolean value that states whether standard scaler has been applied or not

    """
    if "standardScaler" in feat_name:
        return True
    else:
        return False


def get_categoric_pred(feat_names,row_idx, der_fld_idx, model_coef, class_lbls, class_attribute):

    """

    Parameters
    ----------
    feat_names : str
        Contains the name of the field
    row_idx : int
        Contains an integer value to index attribute/column names
    der_fld_idx : int
        Contains an integer value to differentiate between linear and svm models
    model_coef : array
        Contains the estimators coefficient values
    class_lbls : list
        Contains the list of categorical values
    class_attribute : tuple
        Contains Categorical attribute name

    Returns
    -------
    categoric_predictor : list
        Returns a list with instances of nyoka categorical predictor class

    """
    categoric_predictor = list()
    classes_len = len(class_lbls)
    if not is_onehotencoder(feat_names):
        if classes_len == 2:

            if row_idx == -1:
                coef = model_coef
            else:
                coef = model_coef[row_idx][der_fld_idx ]

            cat_pred = pml.CategoricalPredictor(name=class_attribute,
                                                value=class_lbls[-1],
                                                coefficient="{:.16f}".format(coef))
            cat_pred.original_tagname_ = "CategoricalPredictor"
            categoric_predictor.append(cat_pred)
        else:
            for cname, class_idx in zip(class_lbls, range(len(class_lbls))):

                if row_idx == -1:
                    coef = model_coef
                else:
                    coef = model_coef[row_idx][der_fld_idx+class_idx]

                cat_pred = pml.CategoricalPredictor(name=class_attribute,
                                                    value=cname,
                                                    coefficient="{:.16f}".format(coef))
                cat_pred.original_tagname_ = "CategoricalPredictor"
                categoric_predictor.append(cat_pred)
    else:
        for cname, class_idx in zip(class_lbls, range(len(class_lbls))):

            if row_idx == -1:
                coef = model_coef
            else:
                coef = model_coef[row_idx][der_fld_idx + class_idx]

            cat_pred = pml.CategoricalPredictor(name=class_attribute,
                                                value=cname,
                                                coefficient="{:.16f}".format(coef))
            cat_pred.original_tagname_ = "CategoricalPredictor"
            categoric_predictor.append(cat_pred)
    return categoric_predictor




def get_numeric_pred(row_idx, der_fld_idx, model_coef, der_fld_name):
    """

    Parameters
    ----------
    row_idx : int
        Contains an integer value to index attribute/column names
    der_fld_idx : int
        Contains an integer value to differentiate between linear and svm models
    model_coef : array
        Contains the estimators coefficient values
    der_fld_name : string
        Contains the name of the attribute

    Returns
    -------
    num_pred :
        Returns an instances of nyoka numeric predictor class

    """
    num_pred = pml.NumericPredictor(
                        name=der_fld_name,
                        exponent='1',
                        coefficient="{:.16f}".format(model_coef[row_idx][der_fld_idx]))
    num_pred.original_tagname_ = "NumericPredictor"
    return num_pred


