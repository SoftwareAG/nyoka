import pandas as pd
import numpy as np
from nyoka import PMML43Ext as pml
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.preprocessing import Imputer,LabelBinarizer,LabelEncoder,Binarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn_pandas import CategoricalImputer
from sklearn.feature_extraction.text import CountVectorizer
from nyoka.reconstruct.ensemble_tree import reconstruct,Tree
import unicodedata
from nyoka.reconstruct.text import *
import sys
import re
import traceback

def generate_pipeline(parsedPMML,pmmlModel):
    scikitLearnModel = None
    derivedField = list()
    try:
        # nyoka_pmml = pml.parse(pmml, True)
        # pmml_mining_bldtask = nyoka_pmml.MiningBuildTask
        # ext = pmml_mining_bldtask.get_Extension()[0]
        # preProcessingPipeline = ext.get_value()
        preProcessingPipeline = None
        for extension in parsedPMML.get_MiningBuildTask().get_Extension():
            if extension.get_name() == 'preprocessingPipeline':
                preProcessingPipeline = extension.get_value()

        if not preProcessingPipeline:
            return None

        if 'CountVectorizer' in preProcessingPipeline:
            replacement = re.findall('CountVectorizer\((.*)vocabulary=None\)',preProcessingPipeline.replace('\n',''))[0]
            preProcessingPipeline=preProcessingPipeline.replace('CountVectorizer('+replacement+'vocabulary=None)','CountVectorizer()')
        elif 'TfidfVectorizer' in preProcessingPipeline:
            replacement = re.findall('TfidfVectorizer\((.*)vocabulary=None\)',preProcessingPipeline.replace('\n',''))[0]
            preProcessingPipeline=preProcessingPipeline.replace('TfidfVectorizer('+replacement+'vocabulary=None)','TfidfVectorizer()')
        preProcessingPipeline = eval(preProcessingPipeline)

        # if parsedPMML.RegressionModel:
        #     pmmlModel = parsedPMML.RegressionModel[0]
        #     # sk_model_obj = get_regression_model(pmml_modelobj,pmml)
        # elif parsedPMML.NeuralNetwork:
        #     pmmlModel = parsedPMML.NeuralNetwork[0]
        #     # sk_model_obj = get_neural_net_model(nyoka_pmml)
        # elif parsedPMML.TreeModel:
        #     pmmlModel = parsedPMML.TreeModel[0]
        #     # sk_model_obj = get_tree_model(nyoka_pmml)
        # elif parsedPMML.SupportVectorMachineModel:
        #     pmmlModel = parsedPMML.SupportVectorMachineModel[0]
        #     # sk_model_obj = get_svm_model(pmml_modelobj,nyoka_pmml)
        # elif parsedPMML.ClusteringModel:
        #     pmmlModel = parsedPMML.ClusteringModel[0]
        #     # sk_model_obj = get_kmean_model(pmml_modelobj)
        # elif parsedPMML.MiningModel:
        #     pmmlModel = parsedPMML.MiningModel[0]
        #     # sk_model_obj = get_ensemble_model(nyoka_pmml)
        # elif parsedPMML.NaiveBayesModel:
        #     pmmlModel = parsedPMML.NaiveBayesModel[0]
        #     # sk_model_obj = get_naivebayes_model(nyoka_pmml)
        # elif parsedPMML.NearestNeighborModel:
        #     pmmlModel = parsedPMML.NearestNeighborModel[0]
        #     # sk_model_obj = get_knn_model(nyoka_pmml)

        transformationDictionary = parsedPMML.get_TransformationDictionary()
        if transformationDictionary:
            derivedField = transformationDictionary[0].get_DerivedField()
        return getPipelineObject(scikitLearnModel, derivedField, preProcessingPipeline, pmmlModel, parsedPMML)
    except Exception as err:
        print("Error Occurred while reconstructing, details are : {} ".format(str(err)))
        print(str(traceback.format_exc()))

def storePipelineValues(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list, field,
                 attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list, model,*args):
    if args:
        const_exception_list = args[0]
    else:
        const_exception_list = None

    if len(model_list) > 0:
        model_name = model.__class__.__name__
        previous_model_name = model_list[-1].__class__.__name__
        #to handle label binarazer used in sequence in dataframe mapper
        if model.__class__.__name__ == "LabelBinarizer":
            if attr_list_one_pp_obj:
                if field not in attr_list_one_pp_obj:
                    model_name = None
        if model_name != previous_model_name:
            const_one_derivd_fld_vals = list()
            model_list.append(model)
            if attr_list_one_pp_obj:
                attr_list_entire_pp_obj.append(attr_list_one_pp_obj)
            if const_out_list:
                const_one_derivd_fld_vals.append(const_out_list)
            if const_in_list:
                const_one_derivd_fld_vals.append(const_in_list)


            const_merged_list.append(const_one_derivd_fld_vals)
            der_filed_entire_pp.append(der_fld_one_pp_obj)
            const_in_list = list()
            const_out_list = list()
            attr_list_one_pp_obj = list()
            der_fld_one_pp_obj =list()
            if const_out_val is not None:
                const_out_list.append(const_out_val)
            if const_in_val is not None:
                const_in_list.append(const_in_val)
                if model.__class__.__name__ == "PCA":
                    const_in_list = combineAttributes(const_in_list)
            if const_exception_list:
                const_out_list = const_exception_list
                const_exception_list = list()
            attr_list_one_pp_obj.append(field)
            der_fld_one_pp_obj.append(der_fld_name)
            if isinstance(field, list):
                attr_list_one_pp_obj = combineAttributes(attr_list_one_pp_obj)
        else:
            if not model.__class__.__name__ == "PCA":
                if const_in_val is not None:
                    const_in_list.append(const_in_val)
            if not isinstance(const_in_val, list):
                if not model.__class__.__name__ == "LabelBinarizer":
                    attr_list_one_pp_obj.append(field)
            if const_out_val is not None:
                const_out_list.append(const_out_val)
            der_fld_one_pp_obj.append(der_fld_name)



    else:
        model_list.append(model)
        if const_in_val is not None:
            const_in_list.append(const_in_val)
        if const_out_val is not None:
            const_out_list.append(const_out_val)
        if const_exception_list:
            const_out_list = const_exception_list
            const_exception_list = list()

        attr_list_one_pp_obj.append(field)
        der_fld_one_pp_obj.append(der_fld_name)
        if model.__class__.__name__ == "PCA":
            const_in_list = combineAttributes(const_in_list)
        if isinstance(field, list):
            attr_list_one_pp_obj = combineAttributes(attr_list_one_pp_obj)

    return const_out_list, const_in_list, const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,model_list

def getMaxAbsScalerValues(apply_outer, const_out_val, const_out_list, const_in_list, const_merged_list,
                         attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                         model_list):

    field = apply_outer.get_FieldRef()[0]
    field = field.get_field()
    const_in_val = None
    model=MaxAbsScaler()
    pp_components = storePipelineValues(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                 field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                 model_list,model)
    return pp_components

def getMinMaxScalerValues(apply_inner, const_out_val, const_out_list, const_in_list, const_merged_list,
                         attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                         model_list):

    field = apply_inner[0].get_FieldRef()[0]
    field = field.get_field()
    const_in_val = apply_inner[0].get_Constant()[0].get_valueOf_()
    model = MinMaxScaler()
    pp_components = storePipelineValues(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
         field, attr_list_one_pp_obj ,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list, model)

    return pp_components

def getStandardScalerValues(apply_inner,const_out_val,const_out_list,const_in_list,const_merged_list,
                     attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                     model_list):

    field = apply_inner[0].get_FieldRef()[0].get_field()
    const_in_val = apply_inner[0].get_Constant()[0].get_valueOf_()
    model = StandardScaler()
    dframe_components = storePipelineValues(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                     field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list,model)

    return dframe_components

def getRobustScalerValues(apply_inner,const_out_val,const_out_list,const_in_list,const_merged_list,
                      attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                      model_list):

    field = apply_inner[0].get_FieldRef()[0].get_field()
    const_in_val = apply_inner[0].get_Constant()[0].get_valueOf_()
    model = RobustScaler()
    dframe_components = storePipelineValues(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                     field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list,model)
    return dframe_components

def getBinarizerValues(apply_outer, const_out_val, const_out_list, const_in_list,
                                                 const_merged_list, attr_list_one_pp_obj,
                                                 attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list):
    field = apply_outer.get_FieldRef()[0]
    field = field.get_field()
    const_in_val = None
    model = Binarizer()
    pp_components = storePipelineValues(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                 field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                 model_list, model)
    return pp_components

def getPCAValues(apply_outer, const_out_val, const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list):
    field = list()
    const_in_val = list()
    const_out_val = list()
    for apply_element in apply_outer.get_Apply():
        apply_inner = apply_element.get_Apply()
        const_out_val.append(apply_element.get_Constant()[0].get_valueOf_())
        const_in_val.append(apply_inner[0].get_Constant()[0].get_valueOf_())
        field.append(apply_inner[0].get_FieldRef()[0].get_field())
    model = PCA()

    dframe_components = storePipelineValues(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                     field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list, model)
    return dframe_components

def getPolynomialFeaturesValues(apply_outer, const_out_val, const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list):
    apply_inner = apply_outer.get_Apply()
    const_in_val = list()
    field = list()
    for apply_item in apply_inner:
        field.append(apply_item.get_FieldRef()[0].get_field())
        const_in_val.append(apply_item.get_Constant()[0].get_valueOf_())

    model = PolynomialFeatures()
    dframe_components = storePipelineValues(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                     field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list, model)
    return dframe_components

def getMapValueComponents(map_values,const_out_list,const_in_list,const_merged_list,
                            attr_one_pp_obj,attr_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                            model_list):

    const_one_derivd_fld_vals = list()    
    if attr_one_pp_obj:
        attr_entire_pp_obj.append(attr_one_pp_obj)
        attr_one_pp_obj = []
    #attr_one_pp_obj.append(field)
    if const_out_list:
        const_one_derivd_fld_vals.append(const_out_list)
    if const_in_list:
        const_one_derivd_fld_vals.append(const_in_list)
    if const_one_derivd_fld_vals:
        const_merged_list.append(const_one_derivd_fld_vals)
        const_out_list = list()
        const_in_list = list()
        const_one_derivd_fld_vals = list()
    if der_fld_one_pp_obj:
        der_filed_entire_pp.append(der_fld_one_pp_obj)
    der_fld_one_pp_obj.append(der_fld_name)
    model = LabelEncoder()#field_column_pair = map_values.get_FieldColumnPair()[0]
    field = map_values.get_FieldColumnPair()[0].get_field()
    attr_one_pp_obj.append(field)
    attr_entire_pp_obj.append(attr_one_pp_obj)
    attr_one_pp_obj = []
    row = map_values.InlineTable.get_row()                
    main_list = []
    # internal_input = []
    # output = []
    for i in range(len(row)):
        a = []
        for obj_ in row[i].elementobjs_:
            a.append(eval("row[i]." + obj_))
        main_list.append(a)
    for item in main_list:
        const_out_list.append(item[0])
        const_in_list.append(item[1])
    const_one_derivd_fld_vals.append(const_out_list)
    const_one_derivd_fld_vals.append(const_in_list)
    const_merged_list.append(const_one_derivd_fld_vals)
    model_list.append(model)
    const_out_list = list()
    const_in_list = list()
    
    dframe_components = const_out_list,const_in_list,const_merged_list,attr_one_pp_obj,attr_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,model_list

    return dframe_components

def getTfidfVectorizerValues(apply_outer, const_out_list, const_in_list,
                            const_merged_list, attr_list_one_pp_obj,
                            attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list):

    model = TfidfVectorizer()
    txt_index = apply_outer.get_TextIndex()[0]
    if (len(attr_list_entire_pp_obj)!=len(model_list)):
        const_one_pp_obj = []
        model_list.append(model)
        if der_fld_one_pp_obj:
            der_filed_entire_pp.append(der_fld_one_pp_obj)
            der_fld_one_pp_obj = []
        if const_out_list:
            const_one_pp_obj.append(const_out_list)
        if const_in_list:
            const_one_pp_obj.append(const_in_list)
        if const_one_pp_obj:
            const_merged_list.append(const_one_pp_obj)
            const_out_list = []
            const_in_list = []

    der_fld_one_pp_obj.append(der_fld_name)
    const_out_val = apply_outer.get_Constant()[0].get_valueOf_()
    const_in_val = txt_index.get_Extension()[0].get_anytypeobjs_()[0]
    const_out_list.append(const_out_val)
    const_in_list.append(const_in_val)

    dframe_components = const_out_list,const_in_list,const_merged_list , attr_list_one_pp_obj, attr_list_entire_pp_obj ,der_fld_one_pp_obj,der_filed_entire_pp , model_list

    return dframe_components

def getTextIndexValueComponent(der_fld,const_out_list,const_in_list,const_merged_list,
                                attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,
                                der_filed_entire_pp,model_list):
    txt_index = der_fld.get_TextIndex()
    model = CountVectorizer()
    if (len(attr_list_entire_pp_obj)!=len(model_list)):
        const_one_pp_obj = []
        model_list.append(model)
        if der_fld_one_pp_obj:
            der_filed_entire_pp.append(der_fld_one_pp_obj)
            der_fld_one_pp_obj = []
        if const_out_list:
            const_one_pp_obj.append(const_out_list)
        if const_in_list:
            const_one_pp_obj.append(const_in_list)
        if const_one_pp_obj:
            const_merged_list.append(const_one_pp_obj)
            const_out_list = []
            const_in_list = []
    der_fld_one_pp_obj.append(der_fld_name)
    const_out_val = txt_index.get_Constant().get_valueOf_()
    const_in_val = txt_index.get_Extension()[0].get_value()
    const_out_list.append(const_out_val)
    const_in_list.append(const_in_val)

    dframe_components = const_out_list, const_in_list, const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,model_list

    return dframe_components

def getImputerComponents(apply_inner,const_out_val,const_out_list,const_in_list,
                const_merged_list,attr_list_one_pp_obj,
                attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list):
    field = apply_inner[0].get_FieldRef()[0].get_field()
    const_in_val = None
    model = Imputer()
    dframe_components = storePipelineValues(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                     field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list,model)

    return dframe_components

def getApplyComponents(apply_outer,const_out_list,const_in_list,const_merged_list,
                    attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                    model_list):

    func = apply_outer.get_function()
    if apply_outer.get_Constant():
        const_out_val = apply_outer.get_Constant()[0].get_valueOf_()
    else:
        const_out_val = None

    if func == "/":
        apply_inner = apply_outer.get_Apply()
        if apply_inner:
            ext_obj = apply_inner[0].get_Extension()
            if ext_obj:
                ext_obj_val = ext_obj[0].get_anytypeobjs_()
                if "RobustScaler" == ext_obj_val[0]:
                    dframe_components = getRobustScalerValues(apply_inner,const_out_val,const_out_list,const_in_list,
                                                          const_merged_list,attr_list_one_pp_obj,
                                                          attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)

            else:
                dframe_components = getStandardScalerValues(apply_inner,const_out_val,const_out_list,const_in_list,
                                                     const_merged_list,attr_list_one_pp_obj,
                                                     attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
        else:
            dframe_components = getMaxAbsScalerValues(apply_outer,const_out_val,const_out_list,const_in_list,
                                                     const_merged_list,attr_list_one_pp_obj,
                                                     attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif func == "+":
        apply_inner = apply_outer.get_Apply()
        if apply_inner:
            dframe_components = getMinMaxScalerValues(apply_inner,const_out_val,const_out_list,const_in_list,
                                                     const_merged_list,attr_list_one_pp_obj,
                                                     attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif func == "threshold":
        dframe_components = getBinarizerValues(apply_outer,const_out_val, const_out_list, const_in_list,
                                                 const_merged_list, attr_list_one_pp_obj,
                                                 attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list)
    elif func == "sum":
        dframe_components = getPCAValues(apply_outer, const_out_val, const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list)
    elif func == "product":
        dframe_components = getPolynomialFeaturesValues(apply_outer, const_out_val, const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list)
    elif func == "lowercase":
        field = apply_outer.get_FieldRef()[0].get_field()
        if attr_list_one_pp_obj:
            attr_list_entire_pp_obj.append(attr_list_one_pp_obj)
            attr_list_one_pp_obj = []
        attr_list_one_pp_obj.append(field)
        attr_list_entire_pp_obj.append(attr_list_one_pp_obj)
        attr_list_one_pp_obj = []
        dframe_components = const_out_list,const_in_list,const_merged_list ,attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,model_list
    elif func == "*" :
        dframe_components = getTfidfVectorizerValues(apply_outer,const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif func == "if":
        apply_inner = apply_outer.get_Apply()
        dframe_components = getImputerComponents(apply_inner, const_out_val, const_out_list, const_in_list,
                                        const_merged_list, attr_list_one_pp_obj,
                                        attr_list_entire_pp_obj, der_fld_name, der_fld_one_pp_obj, der_filed_entire_pp,
                                        model_list)

    return dframe_components

def getLabelBinarizerValues(norm_descr, const_out_val, const_out_list, const_in_list,
                          const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                          model_list,const_out_exception):
    const_in_val = None
    field = norm_descr.get_field()
    model = LabelBinarizer()
    dframe_components = storePipelineValues(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                     field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list, model,const_out_exception)
    return dframe_components

def getNormalDFComponents(norm_descr, const_out_list, const_in_list, const_merged_list,
                             attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                                   model_list,pmml_data_dict):
    const_out_exception = list()
    const_out_val = norm_descr.get_value()
    id = norm_descr.get_field()
    for data_fld in pmml_data_dict.get_DataField():
        if data_fld.get_name() == id:
            value_obj = data_fld.get_Value()
            if value_obj.__len__() == 2:
                for val in value_obj:
                    const_out_exception.append(val.get_value())

    dframe_components = getLabelBinarizerValues(norm_descr, const_out_val, const_out_list, const_in_list,
                          const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                          model_list,const_out_exception)
    return dframe_components

def getPipelineComponents(der_fld,const_out_list,const_in_list,const_merged_list,
                     attr_one_pp_obj,attr_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,
                     model_list,pmml_data_dict):
    dframe_components = list()
    der_fld_name = der_fld.get_name()
    if der_fld.get_Apply():
        apply_outer = der_fld.get_Apply()
        der_fld_name = der_fld.get_name()
        dframe_components = getApplyComponents(apply_outer,const_out_list,const_in_list,
                                                const_merged_list,attr_one_pp_obj,
                                                attr_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)

    elif der_fld.get_NormDiscrete():
        norm_descr = der_fld.get_NormDiscrete()
        der_fld_name = der_fld.get_name()
        dframe_components = getNormalDFComponents(norm_descr, const_out_list, const_in_list, const_merged_list,
                                                     attr_one_pp_obj, attr_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                                     model_list,pmml_data_dict)

    elif der_fld.get_MapValues():
        map_values = der_fld.get_MapValues()
        der_fld_name = der_fld.get_name()
        dframe_components = getMapValueComponents(map_values,const_out_list,const_in_list,const_merged_list,
                                                    attr_one_pp_obj,attr_entire_pp_obj,
                                                    der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif der_fld.get_TextIndex():
        dframe_components = getTextIndexValueComponent(der_fld,const_out_list,const_in_list,const_merged_list,
                                                            attr_one_pp_obj,attr_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                                            model_list)
    return dframe_components

def getListedPipelines(list_any):
    return [[item] for item in list_any]

def combineAttributes(attributesOfEntirePipeline):
    if attributesOfEntirePipeline:
        if isinstance(attributesOfEntirePipeline[-1], list):
            attributeList = list()
            for attribute in attributesOfEntirePipeline:
                for attr in attribute:
                    if attr not in attributeList:
                        attributeList.append(attr)
            return attributeList
    return attributesOfEntirePipeline

def createDummyDataFrame(attributes,dataDictionary):
    dummyDataList = list()
    for index, attribute in enumerate(attributes):
        if attribute in dataDictionary.keys():
            dataType = dataDictionary[attribute]
            if dataType == "integer":
                dummyDataList.append(np.random.randint(0,100))
            elif dataType == "double":
                dummyDataList.append(np.random.random())
            elif dataType == "string":
                dummyDataList.append("data"+str(index))
    return pd.DataFrame(data=[dummyDataList], columns=attributes)

def create_dummy_dframe(attr_list,data_dict):
    dummy_data_list = list()
    for attr_idx,attr in enumerate(attr_list):
        if attr in data_dict.keys():
            dtype = data_dict[attr]
            if dtype == "integer":
                dummy_data_list.append(np.random.randint(0,100))
            elif dtype == "double":
                dummy_data_list.append(np.random.random())
            elif dtype == "string":
                dummy_data_list.append("data"+str(attr_idx))
    dummy_dframe = pd.DataFrame(data=[dummy_data_list], columns=attr_list)
    return dummy_dframe

def getDataType(dataType):
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
    if 'double' in dataType:
        return np.float64
    if 'integer' in dataType:
        return np.int64
    if 'string' in dataType:
        return str

def storePipelineAttributes(const_merged_list, pp_const_merged_list, pipe, data_dict):
    pp_step_count = 0
    for step in pipe.steps:
        pp_obj = step[1]
        if pp_obj.__class__.__name__ == 'DataFrameMapper':
            dframe_obj = step[1]
            dframe_features = dframe_obj.features
            incrementor = 0
            for dframe_feature_idx, item in zip(range(len(dframe_features)), dframe_features):
                attribute = item[0]
                transformation_obj_list = item[1]
                if not (isinstance(transformation_obj_list,list)):
                    transformation_obj_list = list()
                    transformation_obj_list.append(item[1])
                if isinstance(attribute,list):
                    attribute=attribute[0]
                for transformation_obj_idx,transformation_obj in enumerate(transformation_obj_list):
                    model_type = transformation_obj.__class__.__name__
                    con_item = const_merged_list[incrementor]
                    incrementor+=1
                    dtype = getDataType(data_dict[attribute])
                    if len(con_item) > 1:
                        outer = con_item[0]
                        inner = con_item[1]
                        outer = np.asarray(outer).astype(dtype)
                        inner = np.asarray(inner).astype(dtype)
                    elif len(con_item) == 1:
                        outer = con_item[0]
                        outer = np.asarray(outer).astype(dtype)
                    if "MaxAbsScaler" == model_type:
                        transformation_obj.max_abs_ = outer
                        transformation_obj.scale_ = outer
                    elif "StandardScaler" == model_type:
                        transformation_obj.mean_ = inner
                        transformation_obj.scale_ = outer
                    elif "RobustScaler" == model_type:
                        transformation_obj.scale_ = outer
                        transformation_obj.center_ = inner
                    elif "MinMaxScaler" == model_type:
                        transformation_obj.min_ = outer
                        transformation_obj.scale_ = inner
                    elif "LabelBinarizer" == model_type:
                        transformation_obj.classes_ = outer
                    elif "Binarizer" == model_type:
                        transformation_obj.threshold = outer
                    elif "PCA" == model_type:
                        transformation_obj.mean_ = inner
                        transformation_obj.components_ = outer
                    elif "PolynomialFeatures" == model_type:
                        transformation_obj.powers_ = outer
                    elif "Imputer" == model_type:
                        transformation_obj.statistics_ = outer
                    elif "LabelEncoder" == model_type:
                        transformation_obj.classes_ = outer
                    elif "CountVectorizer" in model_type:
                        feature = {}
                        vocabulary = {}
                        for i in range(len(outer)):
                            feature[outer[i]] = i
                        for item in inner:
                            vocabulary[item] = feature.get(item)
                        transformation_obj.vocabulary_ = vocabulary
                    elif "TfidfVectorizer" in model_type:
                        feature = {}
                        vocabulary = {}
                        feature_list = sorted(inner)
                        for i in range(len(feature_list)):
                            feature[feature_list[i]] = i
                        for item in inner:
                            vocabulary[item] = feature.get(item)
                        transformation_obj.vocabulary_ = vocabulary
                        transformation_obj.__setattr__('idf_',outer)
        else:
            model_type = pp_obj.__class__.__name__
            con_item = pp_const_merged_list[pp_step_count]
            pp_step_count += 1
            if len(con_item) > 1:
                outer = np.asarray(con_item[0]).astype(np.float64)
                inner = np.asarray(con_item[1]).astype(np.float64)
            elif len(con_item) == 1:
                outer = np.asarray(con_item[0]).astype(np.float64)
            if "MaxAbsScaler" == model_type:
                pp_obj.max_abs_ = outer
                pp_obj.scale_ = outer
            elif "StandardScaler" == model_type:
                pp_obj.mean_ = inner
                pp_obj.scale_ = outer
            elif "RobustScaler" == model_type:
                pp_obj.scale_ = outer
                pp_obj.center_ = inner
            elif "MinMaxScaler" == model_type:
                pp_obj.min_ = outer
                pp_obj.scale_ = inner
            elif "LabelBinarizer" == model_type:
                pp_obj.classes_ = outer
            elif "Binarizer" == model_type:
                pp_obj.threshold = outer
            elif "PCA" == model_type:
                pp_obj.mean_ = inner
                pp_obj.components_ = outer
            elif "PolynomialFeatures" == model_type:
                pp_obj.powers_ = outer
            elif "Imputer" == model_type:
                pp_obj.statistics_ = outer
            elif "LabelEncoder" == model_type:
                pp_obj.classes_ = outer
            elif "CountVetorizer" in model_type:
                feature = {}
                vocabulary = {}
                for i in range(len(outer)):
                    feature[outer[i]] = i
                for item in inner:
                    vocabulary[item] = feature.get(item)
                transformation_obj.vocabulary_ = vocabulary
            elif "TfidfVectorizer" in model_type:
                feature = {}
                vocabulary = {}
                feature_list = sorted(inner)
                for i in range(len(feature_list)):
                    feature[feature_list[i]] = i
                for item in inner:
                    vocabulary[item] = feature.get(item)
                transformation_obj.vocabulary_ = vocabulary
                transformation_obj.__setattr__('idf_',outer)         
    return pipe

def storeConstants(outputList,inputList,mergedList):
    tempList = list()
    if outputList:
        tempList.append(outputList)
    if inputList:
        tempList.append(inputList)
    mergedList.append(tempList)
    return mergedList

def isMatchingList(list1,list2):
    return True if all(item in list2 for item in list1) else False

def getImputerValues(miningFields,original_attr_entire_pp_obj,original_model_list):
    const_out_list= list()
    const_merged_list = list()
    attr_list_entire_pp_obj = list()
    model_list = list()
    attr_one_pp_obj = list()
    imputer_pp_components = list()
    for model_idx in range(len(original_model_list)):
        if isinstance(original_model_list[model_idx],list):
            if original_model_list[model_idx][0].__class__.__name__ == "Imputer":
                for miningField in miningFields:
                    const_out_val = miningField.get_missingValueReplacement()
                    field = miningField.get_name()
                    if field in original_attr_entire_pp_obj[model_idx]:
                        attr_one_pp_obj.append(field)
                        const_out_list.append(const_out_val)
                if isMatchingList(original_attr_entire_pp_obj[model_idx],attr_one_pp_obj):
                    model = original_model_list[model_idx][0]
                    const_one_imp__lists = list()
                    model_list.append(model)
                    attr_list_entire_pp_obj.append(attr_one_pp_obj)
                    const_one_imp__lists.append(const_out_list)
                    const_merged_list.append(const_one_imp__lists)
                    attr_one_pp_obj = list()
                    const_out_list = list()
    if model_list:
        model_list = getListedPipelines(model_list)
        imputer_pp_components = const_merged_list,attr_list_entire_pp_obj,model_list
    return imputer_pp_components

def getOriginalPipelineComponents(preProcessingPipeline):
    dataFrameModels = list()
    attributesOfEntirePipeline = list()
    originalPipelineModels = list()
    for step in preProcessingPipeline.steps:
        pipeline = step[1]
        if pipeline.__class__.__name__ == "DataFrameMapper":
            dataFrameMapperFeatures = pipeline.features
            for feature in dataFrameMapperFeatures:
                if not(isinstance(feature[1],list)):
                    dataFrameModels.append([feature[1]])
                else:
                    dataFrameModels.append(feature[1])
                if not(isinstance(feature[0],list)):
                    attributesOfEntirePipeline.append([feature[0]])
                else:
                    attributesOfEntirePipeline.append(feature[0])
        else:
            originalPipelineModels.append([pipeline])
    return attributesOfEntirePipeline, dataFrameModels, originalPipelineModels


def isExceptionModel(scikitLearnModel):
    exceptionModels = ['LinearRegression','LogisticRegression','SVR','SVC']
    return True if scikitLearnModel.__class__.__name__ in exceptionModels else False

def arrangePipelineComponents(sk_model_obj,const_merged_list, fields,attr_entire_pp_obj, model_list,
                                          original_attr_entire_pp_obj, original_df_model_list
                                          ):
    new_const_meregd_list = list()
    new_der_fld_entire_list = list()

    for model_idx, model in enumerate(original_df_model_list):
        for pmml_model_idx, pmml_model_list in enumerate(model_list):
            if len(model) == 1:
                if pmml_model_list[0].__class__.__name__ == model[0].__class__.__name__:
                    if isMatchingList(original_attr_entire_pp_obj[model_idx], attr_entire_pp_obj[pmml_model_idx]):
                        new_const_meregd_list.append(const_merged_list[pmml_model_idx])
                        if not isExceptionModel(sk_model_obj):
                            new_der_fld_entire_list.append(fields[pmml_model_idx])

            elif len(model) > 1:
                models_tuple_len = len(model)
                combined_pmml_model = model_list[pmml_model_idx:pmml_model_idx+models_tuple_len]
                combined_pmml_model = combineAttributes(combined_pmml_model)
                if isObjectsAreEqual(combined_pmml_model,model):
                    if isMatchingList(original_attr_entire_pp_obj[model_idx], attr_entire_pp_obj[pmml_model_idx]):
                        for idx in range(models_tuple_len):
                            new_const_meregd_list.append(const_merged_list[pmml_model_idx+idx])
                        if not isExceptionModel(sk_model_obj):
                            new_der_fld_entire_list.append(fields[pmml_model_idx+idx])

    return new_const_meregd_list, new_der_fld_entire_list

def isObjectsAreEqual(obj1,obj2):
    if isinstance(obj1,list):
        if isinstance(obj2,list):
            for firstObject,secondObject in zip(obj1,obj2):
                if firstObject.__class__.__name__ == secondObject.__class__.__name__:
                    pass
                else:
                    return False
            return True
        else:
            raise TypeError("both items should be a list")
    else:
        if not isinstance(obj2,list):
           if obj1.__class__.__name__ == obj2.__class__.__name__:
               return True
        else:
            return TypeError("wrong data type used")

def getWrappedList(categoricValues):
     return [[value] for value in categoricValues]

def getPredictorValues(categoricFields, dataDictionary):
    predictors = categoricFields.get_CategoricalPredictor()
    categoric_val_one_attr = list()
    categoric_attribute_list = list()
    categoric_val_list = list()
    categoric_model = LabelBinarizer()
    lbl_binarizer_pp_components = tuple()
    const_out_exception = list()
    model_list = list()
    for predictor in predictors:
        categoric_attribute = predictor.get_name()
        categoric_val = predictor.get_value()
        if categoric_attribute not in categoric_attribute_list:
            for data_fld in dataDictionary.get_DataField():
                if data_fld.get_name() == categoric_attribute:
                    value_obj = data_fld.get_Value()
                    if value_obj.__len__() == 2:
                        for val in value_obj:
                            const_out_exception.append(val.get_value())
            if not categoric_attribute_list:
                categoric_val_one_attr.append(categoric_val)
                if const_out_exception:
                    categoric_val_one_attr = const_out_exception
                    const_out_exception = list()
            else:
                categoric_val_list.append(categoric_val_one_attr)
                categoric_val_one_attr = list()
                categoric_val_one_attr.append(categoric_val)
                if const_out_exception:
                    categoric_val_one_attr = const_out_exception
                    const_out_exception = list()

            categoric_attribute_list.append(categoric_attribute)
            model_list.append(categoric_model)
        else:
            categoric_val_one_attr.append(categoric_val)
    categoric_val_list.append(categoric_val_one_attr)
    categoric_val_list = getWrappedList(categoric_val_list)
    if model_list:
        model_list = getListedPipelines(model_list)
        categoric_attribute_list = getListedPipelines(categoric_attribute_list)
        lbl_binarizer_pp_components = categoric_val_list, categoric_attribute_list, model_list

    return lbl_binarizer_pp_components

def getLabelBinarizerExceptionValues(pmmlModel, scikitLearnModel, dataDictionary):
    labelBinarizerPipelineComponents = list()
    if isExceptionModel(scikitLearnModel):
        if scikitLearnModel.__class__.__name__ in ("LinearRegression","LogisticRegression"):
            categoricFields = pmmlModel.get_RegressionTable()[0]
        elif scikitLearnModel.__class__.__name__ in ("SVR","SVC"):
            categoricFields = pmmlModel.get_VectorDictionary().get_VectorFields()
        labelBinarizerPipelineComponents = getPredictorValues(categoricFields,dataDictionary)
    return labelBinarizerPipelineComponents

def segregateParsedItems(*args):
    constant = list()
    attribute = list()
    model = list()
    fields = list()
    for preprocess_obj in args:
        if preprocess_obj:
            constant.extend(preprocess_obj[0])
            attribute.extend(preprocess_obj[1])
            model.extend(preprocess_obj[2])
            if preprocess_obj.__len__() > 3:
                fields.extend(preprocess_obj[3])
            else:
                fields.extend(preprocess_obj[1])

    return constant, attribute, model, fields

def getDataFrameDataTypes(attributes, dataDictionary):
    dataTypeDict = dict()
    dataField = dataDictionary.get_DataField()
    for field in dataField:
        name = field.get_name()
        if name in attributes:
            
            dataTypeDict[name] = field.get_dataType()
    return dataTypeDict

def filterPipe(pipeline):
    step = pipeline.steps[:]
    return Pipeline(steps=step) if step else step

def getPMMLAttributes(miningSchema):
    miningField = miningSchema.get_MiningField()
    return [field.get_name() for field in miningField if field.get_usageType() != 'target']

# by CS
#Temporary Logic
def lowerfirst(x):
    return x[:1].lower()+x[1:]

def getMiningFieldVal(miningField):
    outDict=dict(dict())
    for field in miningField:
        replacementValue = field.get_missingValueReplacement()
        if replacementValue:
            outDict[field.get_name()] = [replacementValue]
    return outDict

def getDerivedFieldVal(derivedField):
    outDict = dict()
    for field in derivedField:
        name = field.get_name()
        firstApply = field.get_Apply()
        MapValues = field.get_MapValues()
        NormDiscrete = field.get_NormDiscrete()
        TextIndex = field.get_TextIndex()
        outputList = []
        inputList = []
        if firstApply:
            if firstApply.get_Constant():
                outDict[name] = [[firstApply.get_Constant()[0].get_valueOf_()]]
                secondApply = firstApply.get_Apply()
                if secondApply:
                    outDict[name].append([secondApply[0].get_Constant()[0].get_valueOf_()])
            else:
                for apply_element in firstApply.get_Apply():
                    if name.startswith("PCA"):
                        apply_inner = apply_element.get_Apply()
                        name = 'pCA('+apply_inner[0].FieldRef[0].get_field()+')'
                        outputList.append(apply_element.get_Constant()[0].get_valueOf_())
                        inputList.append(apply_inner[0].get_Constant()[0].get_valueOf_())
                        outDict[name] = [[outputList],inputList]
                    elif name.startswith('poly'):
                        name = 'polynomialFeatures('+apply_element.FieldRef[0].get_field()+')'
                        if name in outDict.keys():
                            outDict[name][0].append([apply_element.get_Constant()[0].get_valueOf_()])
                        else:
                            outDict[name] = [[[apply_element.get_Constant()[0].get_valueOf_()]]]
        elif MapValues:
            for row in MapValues.InlineTable.get_row():
                a = []
                for obj_ in row.elementobjs_:
                    a.append(eval("row." + obj_))
                outputList.append(a[0])
                inputList.append(a[1])
            outDict[name] = [outputList,inputList]
        elif NormDiscrete:
            value = NormDiscrete.get_value()
            name = name.replace('('+value+')',"")
            if name in outDict.keys():
                outDict[name][0].append(value)
            else:
                outDict[name] = [[value]]
    return outDict

def getPipelineStoreValues(piplineObjects,featureNames,miningField,derivedField):
    impDict = getMiningFieldVal(miningField)
    ppDict = getDerivedFieldVal(derivedField)
    outList = []
    for item, name in zip(piplineObjects,featureNames):
        for stepObject in item:
            if stepObject.__class__.__name__ == "Imputer":
                outList.append([impDict[name]])
            else:
                outList.append(ppDict[lowerfirst(stepObject.__class__.__name__)+'('+name+')'])
    return outList

#!By CS

def getPipelineObject(scikitLearnModel, derivedField, preProcessingPipeline, pmmlModel, parsedPMML):
    constantOutput = list()
    constantInput = list()
    constantMerged = list()
    attributesOfOnePipeline = list()
    attributesOfEntirePipeline = list()
    models = list()
    derivedFieldofOnePipeline = list()
    derivedFieldofEntirePipeline = list()
    fields = list()
    derivedFieldsComibed = list()
    pipelineConstantMerged = None
    originalPipelineComponents = getOriginalPipelineComponents(preProcessingPipeline)
    originalAttributesOfEntirePipeline = originalPipelineComponents[0]
    original_df_models = originalPipelineComponents[1]
    originalPipelineModels = originalPipelineComponents[-1]
    miningSchema = pmmlModel.get_MiningSchema()
    dataDictionary = parsedPMML.get_DataDictionary()
    miningField = miningSchema.get_MiningField()
    labelBinarizer = getLabelBinarizerExceptionValues(pmmlModel,scikitLearnModel,dataDictionary)
    imputer = getImputerValues(miningField, originalAttributesOfEntirePipeline, original_df_models)    #????????????????? Not Working
    for field in derivedField:
        dataFrameComponents = getPipelineComponents(field, constantOutput, constantInput, constantMerged, attributesOfOnePipeline, 
                                            attributesOfEntirePipeline,derivedFieldofOnePipeline,derivedFieldofEntirePipeline,
                                            models,dataDictionary)
        constantOutput = dataFrameComponents[0]
        constantInput = dataFrameComponents[1]
        constantMerged = dataFrameComponents[2]
        attributesOfOnePipeline = dataFrameComponents[3]
        attributesOfEntirePipeline = dataFrameComponents[4]
        derivedFieldofOnePipeline = dataFrameComponents[5]
        derivedFieldofEntirePipeline = dataFrameComponents[6]
        models = dataFrameComponents[7]

    if models:
        constantMerged = storeConstants(constantOutput, constantInput, constantMerged)
        if attributesOfOnePipeline:
            attributesOfEntirePipeline.append(attributesOfOnePipeline)
        models = getListedPipelines(models)
        derivedFieldofEntirePipeline.append(derivedFieldofOnePipeline)
        derivedFieldsComibed = constantMerged, attributesOfEntirePipeline, models, derivedFieldofEntirePipeline

    pmml_extracted_data = segregateParsedItems(labelBinarizer, imputer, derivedFieldsComibed)
    constantMerged = pmml_extracted_data[0]
    attributesOfEntirePipeline = pmml_extracted_data[1]
    models = pmml_extracted_data[2]
    fields = pmml_extracted_data[3]

    original_df_models_len = len(original_df_models)
    constantMerged_len = len(constantMerged)
    if originalPipelineModels:
        if constantMerged_len > original_df_models_len:
            if constantMerged_len > len(originalPipelineModels):
                pipelineConstantMerged = constantMerged[constantMerged_len-len(originalPipelineModels):]
                constantMerged = constantMerged[:constantMerged_len-len(originalPipelineModels)]
            elif constantMerged_len == len(originalPipelineModels):
                pipelineConstantMerged = constantMerged
    arranged_data = arrangePipelineComponents(scikitLearnModel, constantMerged, fields, attributesOfEntirePipeline, models,
                                              originalAttributesOfEntirePipeline, original_df_models)                         
    # constantMerged = arranged_data[0]
    # if not originalPipelineModels:
    #     fields = arranged_data[1]
    # else:
    #     # sub_len = len(originalPipelineModels)-1
    #     fields = fields[-1]
    if not originalAttributesOfEntirePipeline:
        originalAttributesOfEntirePipeline = getPMMLAttributes(miningSchema)
    attr_list = combineAttributes(originalAttributesOfEntirePipeline)
    fields = combineAttributes(fields)

    constantMerged = getPipelineStoreValues(original_df_models, attr_list, miningField, derivedField)
    pipe = filterPipe(preProcessingPipeline)
    if pipe:
        data_dict = getDataFrameDataTypes(attr_list, dataDictionary)
        dummy_dframe = createDummyDataFrame(attr_list, data_dict)
        pipe = pipe.fit(dummy_dframe)
        pipe = storePipelineAttributes(constantMerged, pipelineConstantMerged, pipe, data_dict)
    else:
        pass
    return pipe