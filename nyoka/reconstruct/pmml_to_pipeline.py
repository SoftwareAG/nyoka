import pandas as pd
import numpy as np
from nyoka import PMML43Ext as pml
# from PMML43Ext import row
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.preprocessing import Imputer,LabelBinarizer,LabelEncoder,Binarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn_pandas import CategoricalImputer
#from text import *
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVR,SVC, LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from nyoka.reconstruct.ensemble_tree import reconstruct,Tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor,IsolationForest
import unicodedata
from nyoka.reconstruct.text import *

import sys
import re
import traceback


def generate_skl_pipeline(pmml):
    sk_model_obj = None
    der_field = list()
    try:
        nyoka_pmml = pml.parse(pmml, True)
        pmml_mining_bldtask = nyoka_pmml.MiningBuildTask
        ext = pmml_mining_bldtask.get_Extension()[0]
        pipeline_obj = ext.get_value()
        import re
        if 'CountVectorizer' in pipeline_obj:
            replace_obj=re.findall('CountVectorizer\((.*)vocabulary=None\)',pipeline_obj.replace('\n',''))[0]
            pipeline_obj=pipeline_obj.replace('CountVectorizer('+replace_obj+'vocabulary=None)','CountVectorizer()')
        elif 'TfidfVectorizer' in pipeline_obj:
            replace_obj=re.findall('TfidfVectorizer\((.*)vocabulary=None\)',pipeline_obj.replace('\n',''))[0]
            pipeline_obj=pipeline_obj.replace('TfidfVectorizer('+replace_obj+'vocabulary=None)','TfidfVectorizer()')
        pipeline_obj = eval(pipeline_obj)

        if nyoka_pmml.RegressionModel:
            pmml_modelobj = nyoka_pmml.RegressionModel[0]
            sk_model_obj = get_regression_model(pmml_modelobj,pmml)
        elif nyoka_pmml.NeuralNetwork:
            pmml_modelobj = nyoka_pmml.NeuralNetwork[0]
            sk_model_obj = get_neural_net_model(nyoka_pmml)
        elif nyoka_pmml.TreeModel:
            pmml_modelobj = nyoka_pmml.TreeModel[0]
            sk_model_obj = get_tree_model(nyoka_pmml)
        elif nyoka_pmml.SupportVectorMachineModel:
            pmml_modelobj = nyoka_pmml.SupportVectorMachineModel[0]
            sk_model_obj = get_svm_model(pmml_modelobj,nyoka_pmml)
        elif nyoka_pmml.ClusteringModel:
            pmml_modelobj = nyoka_pmml.ClusteringModel[0]
            sk_model_obj = get_kmean_model(pmml_modelobj)
        elif nyoka_pmml.MiningModel:
            pmml_modelobj = nyoka_pmml.MiningModel[0]
            sk_model_obj = get_ensemble_model(nyoka_pmml)
        elif nyoka_pmml.NaiveBayesModel:
            pmml_modelobj = nyoka_pmml.NaiveBayesModel[0]
            sk_model_obj = get_naivebayes_model(nyoka_pmml)
        elif nyoka_pmml.NearestNeighborModel:
            pmml_modelobj = nyoka_pmml.NearestNeighborModel[0]
            sk_model_obj = get_knn_model(nyoka_pmml)

        loc_transform = nyoka_pmml.get_TransformationDictionary()
        if loc_transform:
            der_field = loc_transform[0].get_DerivedField()
        sk_model_obj = get_pipelineobj(sk_model_obj, der_field, pipeline_obj, pmml_modelobj, nyoka_pmml)
        return sk_model_obj
    except Exception as err:
        print("Error Occurred while reconstructing, details are : {} ".format(str(err)))
        print(str(traceback.format_exc()))


def get_knn_model(pmml):
    knn_model = pmml.NearestNeighborModel[0]
    func_name = knn_model.get_functionName()
    y = list()
    data = list()
    rows = knn_model.get_TrainingInstances().InlineTable.get_row()
    for row_idx, item in enumerate(rows):
        attribute_value = list()
        for obj_ in item.elementobjs_:
            attribute_value.append(eval('item.' + obj_))
        data.append(attribute_value)
        y.append(data[row_idx].pop(0))

    data = np.array(data)
    y = np.array(y)

    p_value = 0
    if knn_model.ComparisonMeasure.get_cityBlock():
        p_value = 1
    elif knn_model.ComparisonMeasure.get_euclidean():
        p_value = 2
    elif knn_model.ComparisonMeasure.get_minkowski():
        p_value = 3
    n_neigh = knn_model.numberOfNeighbors

    if func_name == 'classification':
        sk_model_obj = KNeighborsClassifier()
        sk_model_obj.p = p_value
        sk_model_obj.n_neighbors = n_neigh
    elif func_name == 'regression':
        sk_model_obj = KNeighborsRegressor()
        sk_model_obj.p = p_value
        sk_model_obj.n_neighbors = n_neigh
    else:
        raise ValueError("Function name in the pmml is not identified")

    sk_model_obj.fit(data, y)
    return sk_model_obj



def get_naivebayes_model(pmml):

    gnb_model = pmml.NaiveBayesModel[0]
    funct_name = gnb_model.get_functionName()
    bs_inputs_obj = gnb_model.get_BayesInputs()
    bs_inputs = bs_inputs_obj.get_BayesInput()
    bs_output_obj = gnb_model.get_BayesOutput()
    output_counts_obj = bs_output_obj.get_TargetValueCounts()
    output_counts = output_counts_obj.get_TargetValueCount()
    class_count = list()
    classes = list()
    theta_lst = list()
    sigma_lst = list()
    class_prior = list()

    for b_input in bs_inputs:
        means_lst = []
        variance_lst = []
        tr_val_stats_obj = b_input.get_TargetValueStats()
        if tr_val_stats_obj:
            tr_val_stats = tr_val_stats_obj.get_TargetValueStat()
            for tr_val in tr_val_stats:
                if tr_val.PoissonDistribution:
                    gsd = tr_val.get_PoissonDistribution()
                elif tr_val.GaussianDistribution:
                    gsd = tr_val.get_GaussianDistribution()
                elif tr_val.UniformDistribution:
                    gsd = tr_val.UniformDistribution
                else:
                    return TypeError("Unknown TargetValueStat type in Nyoka")
                means_lst.append(gsd.get_mean())
                variance_lst.append(gsd.get_variance())
        if means_lst:
            theta_lst.append(means_lst)
        if variance_lst:
            sigma_lst.append(variance_lst)

    for output in output_counts:
        if is_number(output.get_value()):
            classes.append(int(output.get_value()))
        else:
            classes.append(output.get_value())
        class_count.append(output.get_count())

    theta_lst = list(map(list, zip(*theta_lst)))
    sigma_lst = list(map(list, zip(*sigma_lst)))

    count_total = np.sum(class_count)
    for count in class_count:
        prior = count / count_total
        class_prior.append(prior)

    if funct_name == "classification":
        sk_model_obj = GaussianNB()
        sk_model_obj.theta_ = np.array(theta_lst, dtype='float64')
        sk_model_obj.sigma_ = np.array(sigma_lst, dtype='float64')
        sk_model_obj.class_count_ = np.array(class_count,
                                             dtype='float64')
        sk_model_obj.classes_ = np.array(classes)
        sk_model_obj.class_prior_ = np.array(class_prior, dtype='float64')
    else:
        raise Exception("Pmml model function name is not : classification")

    return sk_model_obj




def get_ensemble_model(pmml,*args):
    recon_model = reconstruct(pmml,*args)
    return recon_model


def get_neural_net_model(pmml):
    model = pmml.NeuralNetwork[0]
    algorithm_names = model.get_algorithmName()
    activation_functions = model.get_activationFunction()
    if activation_functions == 'rectifier':
        activation_functions = 'relu'
    function_names = model.get_functionName()
    thresholds = model.get_threshold()
    altitudes = model.get_altitude()
    no_of_layers = model.get_numberOfLayers()

    # Fetching bias and coefficient values
    # Fetching bias and coefficient values
    biass = []
    for j in range(len(model.get_NeuralLayer()) - 2):
        biass.append([])
        for i in range(len(model.NeuralLayer[j].get_Neuron())):
            biass[j].append(float(model.NeuralLayer[j].Neuron[i].get_bias()))
    bias = []
    for i in range(len(biass)):
        bias.append(np.array(biass[i]))
    coefs = []
    for j in range(len(model.get_NeuralLayer()) - 2):
        coefs.append([])

        for i in range(len(model.NeuralLayer[j].Neuron[0].get_Con())):
            coefs[j].append([])

            for k in range(len(model.NeuralLayer[j].get_Neuron())):
                coefs[j][i].append(model.NeuralLayer[j].Neuron[k].Con[i].get_weight())
    coef = []
    for i in range(len(coefs)):
        coef.append(np.array(coefs[i]))

    # Hidden Layer sizes
    l = []
    hidden_layers = len(model.get_NeuralLayer()) - 3
    for i in range(hidden_layers):
        l.append(len(model.NeuralLayer[i].get_Neuron()))
    outputs = len(model.get_NeuralOutputs().get_NeuralOutput())

    # Creating NN Model
    if function_names == "regression":
        sk_model = MLPRegressor()
        sk_model.activation = activation_functions
        sk_model.hidden_layer_sizes = tuple(l)
        sk_model.intercepts_ = bias
        sk_model.coefs_ = coef
        sk_model.noOfLayer = hidden_layers
        sk_model.n_layers_ = len(hidden_layers) + 2


    elif function_names == "classification":
        sk_model = MLPClassifier()
        sk_model.activation = activation_functions
        sk_model.hidden_layer_sizes = tuple(l)
        sk_model.intercepts_ = bias
        sk_model.coefs_ = coef
        sk_model.n_outputs_ = 1 if outputs == 2 else outputs
        sk_model.noOfLayer = hidden_layers
        sk_model.n_layers_ = len(model.get_NeuralLayer()) - 1
        sk_model.out_activation_ = 'logistic'
        sk_model._label_binarizer = LabelBinarizer()

    return sk_model

def get_kmean_model(pmml_modelobj):
    funct_name = pmml_modelobj.get_functionName()
    cluster_num = pmml_modelobj.get_numberOfClusters()
    centroids = []
    centers = []
    for i in range(cluster_num):
        clust = pmml_modelobj.get_Cluster()[i]
        arr_val = clust.get_Array()
        val = arr_val.content_[0].value
        centroids = val.split()
        for j in range(len(centroids)):
            centers.append(centroids[j])
    center_matrix = np.asarray(centers, dtype="float64").reshape(cluster_num, len(centroids))
    if funct_name == "clustering":
        sk_model_obj = KMeans(n_clusters=cluster_num, random_state=0)
        sk_model_obj.cluster_centers_ = center_matrix
    else:
        raise Exception("Pmml model function name is not : regression")
    return sk_model_obj


def get_svm_model(svm, pmml):
    funct_name = svm.functionName
    svm_model = svm
    data_dict = pmml.get_DataDictionary()
    data_flds = data_dict.get_DataField()
    class_cnt = 0
    classes = []
    supports = []
    support = []
    coefs1 = []
    vec_id = []  # support_
    val_coef = []
    intercept_f = []
    coefs2 = []

    if funct_name == "classification":
        for val in data_flds[-1].get_Value():
            classes.append(val.get_value())
        class_cnt=len(classes)
        svms = svm.get_SupportVectorMachine()
        if len(classes) > 2:
            set_of_vectors = [set() for _ in range(class_cnt)]
            matrix = [[[] for _ in range(class_cnt)] for _ in range(class_cnt)]
            for svm in svms:
                # svm.export(sys.stdout,0,"")
                target = svm.get_targetCategory()
                alter = svm.get_alternateTargetCategory()
                sup_vecs = svm.get_SupportVectors()
                for sup_vec in sup_vecs.get_SupportVector():
                    matrix[classes.index(target)][classes.index(alter)].append(int(sup_vec.get_vectorId()))

            flattened = []
            for i in range(class_cnt):
                for j in range(class_cnt):
                    if len(matrix[j][i]) > 0:
                        flattened.append(set(matrix[j][i]))
            cnt = 0
            for i in range(0, class_cnt):
                temp = set()
                for j in range(i + 1, class_cnt):
                    if len(temp) == 0:
                        temp = set(matrix[j][i])
                    else:
                        temp = temp & set(matrix[j][i])
                for j in range(i - 1, -1, -1):
                    if len(temp) == 0:
                        temp = set(matrix[i][j])
                    else:
                        temp = temp & set(matrix[i][j])
                set_of_vectors[cnt] = temp
                cnt += 1

            n_support = [len(a) for a in set_of_vectors]

            support = sorted([a for _ in set_of_vectors for a in _])
            dual_coef = []
            dual_coefs = [[[] for _ in range(class_cnt)] for _ in range(class_cnt - 1)]
            intercept = []
            for svm in svms:
                target = svm.get_targetCategory()
                alter = svm.get_alternateTargetCategory()
                coefs = svm.get_Coefficients()
                intercept.append(coefs.get_absoluteValue())
                cnt = 0
                lst1 = []
                lst2 = []
                for coef in coefs.get_Coefficient():
                    if cnt < n_support[classes.index(alter)]:
                        lst1.append(coef.get_value())
                        cnt += 1
                    else:
                        lst2.append(coef.get_value())
                dual_coefs[classes.index(target) - 1][classes.index(alter)] = lst1
                dual_coefs[classes.index(alter)][classes.index(target)] = lst2

            for i in range(len(dual_coefs)):
                val_list = []
                for lst in dual_coefs[i]:
                    for val in lst:
                        val_list.append(val)
                dual_coef.append(val_list)

        else:
            svms = svms[0]
            sup_vec = svms.get_SupportVectors()
            support = []
            prev = -1
            cl1_cnt = 0
            cl2_cnt = 0
            for vec in sup_vec.get_SupportVector():
                if int(vec.get_vectorId()) < prev:
                    cl2_cnt += 1
                else:
                    cl1_cnt += 1
                    prev = int(vec.get_vectorId())
                support.append(int(vec.get_vectorId()))

            n_support = [cl1_cnt, cl2_cnt]

            intercept = []
            dual_coef = []
            coefs = svms.get_Coefficients()
            intercept.append(coefs.get_absoluteValue())
            for coe in coefs.get_Coefficient():
                dual_coef.append(coe.get_value())

        vect_dict = svm_model.VectorDictionary
        vec_dic = []
        vec_dic.append(svm_model.get_VectorDictionary())
        vfs = vec_dic[0].get_VectorFields()
        field_ref = vfs.get_FieldRef()
        cat_pred = vfs.get_CategoricalPredictor()
        input_shape = (-1,len(field_ref) + len(cat_pred))

        no_of_vectors = vect_dict.numberOfVectors
        vec_fields = vect_dict.VectorFields
        no_of_fields = vec_fields.numberOfFields
        vect_ins = vect_dict.VectorInstance

        support_vectors = []
        for ins in vect_ins:
            entry = []
            real_entry = ins.get_REAL_SparseArray().get_REAL_Entries()
            for val in real_entry:
                entry.append(float(val))
            support_vectors.append(entry)

        linear = svm_model.get_LinearKernelType()
        poly = svm_model.get_PolynomialKernelType()
        rbf = svm_model.get_RadialBasisKernelType()

    elif funct_name == "regression":
        supports.append(svm.get_SupportVectorMachine()[0].get_SupportVectors())
        for supports_idx in range(len(supports)):
            vecs = []
            support.append(supports[0].get_SupportVector())

            for supports_elem_idx in range(len(support[supports_idx])):
                vecs.append(support[supports_idx][supports_elem_idx].get_vectorId())

            vec_id.append(vecs)

        # coefficient calculation
        coefs1.append(svm.get_SupportVectorMachine()[0].get_Coefficients())
        for c1 in range(len(coefs1)):
            intcpt = []
            # the absolute values in coefficients
            intcpt.append(coefs1[0].get_absoluteValue())
            val = []
            coefs2.append(coefs1[0].get_Coefficient())
            # getting the values of the coefficients
            for c2 in range(len(coefs2[c1])):
                val.append(coefs2[c1][c2].get_value())
            val_coef.append(val)
        intercept_f.append(intcpt)
        classes.append(0)

        # vector instances and real entries
        vec_dic = []
        vec_ins = []
        entries = []  # support_vectors_
        vec_dic.append(svm_model.get_VectorDictionary())
        vfs = vec_dic[0].get_VectorFields()
        field_ref = vfs.get_FieldRef()
        cat_pred = vfs.get_CategoricalPredictor()
        input_shape = (-1,len(field_ref) + len(cat_pred))
        for i in range(len(vec_dic)):
            vec_ins.append(vec_dic[i].get_VectorInstance())
            for j in range(len(vec_ins[0])):
                entries.append(vec_ins[i][j].get_REAL_SparseArray().get_REAL_Entries())

        # converting string to list
        supp_ = [list(map(int, x)) for x in vec_id]
        supp_vec = [list(map(float, y)) for y in entries]

        aa = set(supp_[0])
        np_classes = np.array(classes)
        np_sup_vec = np.array(supp_vec)
        np_intercept = np.array(intercept_f)
        np_support_ = np.array(supp_)
        np_duals = np.array(val, dtype='float64')
        np_n_supp = np.array(aa)
        proba = []
        np_proba = np.array(proba)
        probB = []
        np_probB = np.array(probB)
        linear = svm.get_LinearKernelType()
        poly = svm.get_PolynomialKernelType()
        rbf = svm.get_RadialBasisKernelType()

    if linear:
        if funct_name == "regression":
            model_obj = SVR(kernel='linear')
        elif funct_name == "classification":
            model_obj = SVC(kernel='linear')
        model_obj._gamma = 0.0
    if poly:
        if funct_name == "regression":
            model_obj = SVR(kernel='poly', degree=int(poly.get_degree()))
        elif funct_name == "classification":
            model_obj = SVC(kernel='poly', degree=int(poly.get_degree()))
        model_obj._gamma = poly.get_gamma()

    if rbf:
        if funct_name == "regression":
            model_obj = SVR(kernel='rbf')
        elif funct_name == "classification":
            model_obj = SVC(kernel='rbf')
        model_obj._gamma = rbf.get_gamma()

    if funct_name == "regression":
        model_obj.support_vectors_ = np_sup_vec
        model_obj._intercept_ = np_intercept.reshape(1, )
        model_obj.support_ = np_support_.reshape(-1, )
        model_obj._dual_coef_ = np_duals.reshape(1, -1)
        model_obj.dual_coef_ = model_obj._dual_coef_
        model_obj._sparse = False
        model_obj.shape_fit_ = input_shape
        model_obj.n_support_ = np.asarray([0, 0])
        model_obj.probA_ = np_proba
        model_obj.probB_ = np_probB

    elif funct_name == "classification":
        if data_flds[-1].get_dataType() == 'integer':
            classes = [int(clas) for clas in classes]
        model_obj.support_vectors_ = np.asarray(support_vectors)
        if len(classes) > 2:
            model_obj.classes_ = np.asarray(classes)
        else:
            classes.reverse()
            model_obj.classes_ = np.asarray(classes)
        model_obj._intercept_ = np.asarray(intercept)
        model_obj.support_ = np.asarray(support)
        model_obj._dual_coef_ = np.asarray(dual_coef).reshape(-1, len(support))
        model_obj.dual_coef_ = model_obj._dual_coef_
        model_obj._sparse = False
        model_obj.shape_fit_ = input_shape
        model_obj.n_support_ = np.asarray(n_support)
        model_obj.probA_ = np.asarray([])
        model_obj.probB_ = np.asarray([])

    return model_obj



def get_predictor_map(regr_table_items):
    regr_predictor_dict = dict()
    idx_counter = 0
    for single_predictor in regr_table_items:
        if "NumericPredictor" in single_predictor:
            regr_predictor_dict[idx_counter] = "NumericPredictor"
            idx_counter += 1
        if "CategoricalPredictor" in single_predictor:
            regr_predictor_dict[idx_counter] = "CategoricalPredictor"
            idx_counter += 1
    return regr_predictor_dict


# def insert_coef_vals(num_predictors, cat_predictors, regr_predictor_dict):
#     coefs = list()
#     for value,key in regr_predictor_dict.items():
#         if key == "NumericPredictor":
#             pred = num_predictors[0]
#             coefs.append(float(pred.get_coefficient()))
#             num_predictors = num_predictors[1:]
#         if key == "CategoricalPredictor":
#             pred = cat_predictors[0]
#             coefs.append(float(pred.get_coefficient()))
#             cat_predictors = cat_predictors[1:]
#     return coefs


def insert_coef_vals(num_predictors, cat_predictors, regr_predictor_dict):
    coefs = list()
    if regr_predictor_dict:
        for value,key in regr_predictor_dict.items():
            if key == "NumericPredictor":
                pred = num_predictors[0]
                coefs.append(float(pred.get_coefficient()))
                num_predictors = num_predictors[1:]
            if key == "CategoricalPredictor":
                pred = cat_predictors[0]
                coefs.append(float(pred.get_coefficient()))
                cat_predictors = cat_predictors[1:]
    return coefs


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_regression_model(cls_model, pmml):
    cat_predictors = list()
    num_predictors = list()
    funct_name = cls_model.get_functionName()
    model = cls_model.modelName
    regression_table = cls_model.get_RegressionTable()
    pmml_file = open(pmml, "r")
    pmml_file_data = pmml_file.read()
    substring1 = "RegressionTable"
    substring2 = "/RegressionTable"
    substr_idx = 0
    regr_table_list = list()
    coefs = list()
    classes = list()
    intercept = list()
    for reg_idx,reg_table in enumerate(regression_table):
        cat_predictors = list()
        num_predictors = list()
        regr_predictor_dict = dict()
        try:
            regr_table_str = pmml_file_data[(pmml_file_data.index(substring1) + len(substring1)):pmml_file_data.index(substring2)]
            substr_idx = pmml_file_data.index(substring2) + len(substring2)
            pmml_file_data = pmml_file_data[substr_idx:]
        except ValueError:
            regr_table_str = ''
        if regr_table_str:
            regr_table_items = regr_table_str.split('\n')
            regr_predictor_dict = get_predictor_map(regr_table_items)
            if reg_table.get_NumericPredictor():
                num_predictors = reg_table.get_NumericPredictor()
            if reg_table.get_CategoricalPredictor():
                cat_predictors = reg_table.get_CategoricalPredictor()
            if reg_table.get_targetCategory():
                if (is_number(reg_table.get_targetCategory())):
                    classes.append(int(reg_table.get_targetCategory()))
                else:
                    classes.append(reg_table.get_targetCategory())
        else:
            if reg_table.get_targetCategory():
                if (is_number(reg_table.get_targetCategory())):
                    classes.insert(0,int(reg_table.get_targetCategory()))
                else:
                    classes.insert(0,reg_table.get_targetCategory())
        coef_single_reg_tbl = insert_coef_vals(num_predictors, cat_predictors, regr_predictor_dict)
        if coef_single_reg_tbl:
            coefs.append(np.asarray(coef_single_reg_tbl,np.float64))
            intercept.append(float(reg_table.get_intercept()))

    coefs = np.asarray(coefs, dtype=np.float64)
    classes = np.asarray(classes, dtype=object)
    intercept = np.asarray(intercept, dtype=np.float64)
    if funct_name == "regression":
        sk_model_obj = LinearRegression()
        sk_model_obj.intercept_ = intercept[0]
        sk_model_obj.coef_ = np.asarray(coefs[0], dtype=np.float64)
    elif funct_name == "classification":
        if model == 'RidgeClassifier':
            sk_model_obj = linear_model.RidgeClassifier()
        elif model == 'SGDClassifier':
            sk_model_obj = linear_model.SGDClassifier()
        elif model == 'LinearDiscriminantAnalysis':
            sk_model_obj = LinearDiscriminantAnalysis()
        else:
            sk_model_obj = linear_model.LogisticRegression()
        sk_model_obj.intercept_ = intercept
        sk_model_obj.coef_ = coefs
        if model == 'RidgeClassifier':
            from sklearn.preprocessing import LabelBinarizer
            sk_model_obj._label_binarizer = LabelBinarizer(neg_label=-1, pos_label=1,sparse_output=False)
            sk_model_obj._label_binarizer.classes_ = np.array(classes)
            sk_model_obj._label_binarizer.sparse_input_ = False
            if len(classes)==2:
                sk_model_obj._label_binarizer.y_type_ = 'binary'
            else:
                sk_model_obj._label_binarizer.y_type_ = 'multiclass'
        else:
            sk_model_obj.classes_ = classes
    else:
        raise Exception("Pmml model function name is not : regression")
    return sk_model_obj


####Tree Reconstruction---------Start####

def assign_fields(der_field, mining_flds):
    fields = list()
    for der_fld_idx, one_der_fld in enumerate(der_field):
        if one_der_fld.get_Apply():
            apply_fld = one_der_fld.get_Apply()
            if hasattr(apply_fld, 'get_function'):
                if apply_fld.get_function() == 'lowercase':
                    del der_field[der_fld_idx]
    for fld in der_field:
        fields.append(fld.name)
    return fields


def get_data_information(pmml):
    tree_model = pmml.get_TreeModel()[0]
    mining = tree_model.get_MiningSchema()
    mfs = mining.get_MiningField()
    fields = list()
    classes = list()
    for mm in mfs:
        if mm.get_usageType() != 'target':
            fields.append(mm.get_name())
        else:
            target_name = mm.get_name()
            dt = pmml.get_DataDictionary()
            for dd in dt.get_DataField():
                if dd.get_name() == target_name:
                    val = dd.get_Value()
                    for vv in val:
                        if dd.get_dataType() == 'integer':
                            classes.append(int(vv.get_value()))
                        else:
                            classes.append(vv.get_value())

    return classes, fields


def get_tree_model(pmml,*args):
    tree_model = pmml.get_TreeModel()[0]
    func_name = tree_model.get_functionName()
    main_node = tree_model.get_Node()
    all_node = main_node.get_Node()
    classes, fields = get_data_information(pmml)
    if args:
        classes = get_data_information(pmml)[0]
        fields = args[0]
    if func_name == 'regression':
        model = DecisionTreeRegressor()
    else:
        model = DecisionTreeClassifier()
    model.n_features = len(fields)
    model.n_features_ = len(fields)
    model.n_outputs_ = 1
    model.n_outputs = 1
    model.classes_ = np.array(classes)
    model.n_classes_ = len(classes)
    model._estimator_type = 'classifier' if len(classes) > 0 else 'regressor'
    tree = Tree(fields, classes)
    tree.get_node_info(all_node)
    tree.build_tree()
    model.tree_ = tree
    return model


####----------End----------------####


def store_pp_val(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list, field,
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
                    const_in_list = combine_attributes(const_in_list)
            if const_exception_list:
                const_out_list = const_exception_list
                const_exception_list = list()
            attr_list_one_pp_obj.append(field)
            der_fld_one_pp_obj.append(der_fld_name)
            if isinstance(field, list):
                attr_list_one_pp_obj = combine_attributes(attr_list_one_pp_obj)
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
            const_in_list = combine_attributes(const_in_list)
        if isinstance(field, list):
            attr_list_one_pp_obj = combine_attributes(attr_list_one_pp_obj)

    return const_out_list, const_in_list, const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,model_list


def get_max_abs_sclr_val(apply_outer, const_out_val, const_out_list, const_in_list, const_merged_list,
                         attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                         model_list):

    field = apply_outer.get_FieldRef()[0]
    field = field.get_field()
    const_in_val = None
    model=MaxAbsScaler()
    pp_components = store_pp_val(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                 field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                 model_list,model)
    return pp_components


def get_min_max_sclr_val(apply_inner, const_out_val, const_out_list, const_in_list, const_merged_list,
                         attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                         model_list):

    field = apply_inner[0].get_FieldRef()[0]
    field = field.get_field()
    const_in_val = apply_inner[0].get_Constant()[0].get_valueOf_()
    model = MinMaxScaler()
    pp_components = store_pp_val(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
         field, attr_list_one_pp_obj ,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list, model)

    return pp_components


def get_std_sclr_val(apply_inner,const_out_val,const_out_list,const_in_list,const_merged_list,
                     attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                     model_list):

    field = apply_inner[0].get_FieldRef()[0].get_field()
    const_in_val = apply_inner[0].get_Constant()[0].get_valueOf_()
    model = StandardScaler()
    dframe_components = store_pp_val(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                     field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list,model)

    return dframe_components


def get_rbst_sclr_val(apply_inner,const_out_val,const_out_list,const_in_list,const_merged_list,
                      attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                      model_list):

    field = apply_inner[0].get_FieldRef()[0].get_field()
    const_in_val = apply_inner[0].get_Constant()[0].get_valueOf_()
    model = RobustScaler()
    dframe_components = store_pp_val(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                     field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list,model)
    return dframe_components


def get_binarizer_val(apply_outer, const_out_val, const_out_list, const_in_list,
                                                 const_merged_list, attr_list_one_pp_obj,
                                                 attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list):
    field = apply_outer.get_FieldRef()[0]
    field = field.get_field()
    const_in_val = None
    model = Binarizer()
    pp_components = store_pp_val(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                 field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                 model_list, model)
    return pp_components


def get_pca_val(apply_outer, const_out_val, const_out_list, const_in_list,
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

    dframe_components = store_pp_val(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                     field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list, model)
    return dframe_components


def get_poly_feat_val(apply_outer, const_out_val, const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list):
    apply_inner = apply_outer.get_Apply()
    const_in_val = list()
    field = list()
    for apply_item in apply_inner:
        field.append(apply_item.get_FieldRef()[0].get_field())
        const_in_val.append(apply_item.get_Constant()[0].get_valueOf_())

    model = PolynomialFeatures()
    dframe_components = store_pp_val(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                     field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list, model)
    return dframe_components


def get_mapValueComponents(map_values,const_out_list,const_in_list,const_merged_list,
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
    internal_input = []
    output = []
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


def get_tfidf_val(apply_outer, const_out_list, const_in_list,
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

def get_textIndexValueComponent(der_fld,const_out_list,const_in_list,const_merged_list,
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
    const_in_val = txt_index.get_Extension()[0].get_anytypeobjs_()[0]
    const_out_list.append(const_out_val)
    const_in_list.append(const_in_val)

    dframe_components = const_out_list, const_in_list, const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,model_list

    return dframe_components


def get_imp_val(apply_inner,const_out_val,const_out_list,const_in_list,
                const_merged_list,attr_list_one_pp_obj,
                attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list):
    field = apply_inner[0].get_FieldRef()[0].get_field()
    const_in_val = None
    model = Imputer()
    dframe_components = store_pp_val(const_out_val,const_out_list,const_in_val,const_in_list,const_merged_list,
                                     field,attr_list_one_pp_obj,attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list,model)

    return dframe_components


def get_applycomponents(apply_outer,const_out_list,const_in_list,const_merged_list,
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
                    dframe_components = get_rbst_sclr_val(apply_inner,const_out_val,const_out_list,const_in_list,
                                                          const_merged_list,attr_list_one_pp_obj,
                                                          attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)

            else:
                dframe_components = get_std_sclr_val(apply_inner,const_out_val,const_out_list,const_in_list,
                                                     const_merged_list,attr_list_one_pp_obj,
                                                     attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
        else:
            dframe_components = get_max_abs_sclr_val(apply_outer,const_out_val,const_out_list,const_in_list,
                                                     const_merged_list,attr_list_one_pp_obj,
                                                     attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif func == "+":
        apply_inner = apply_outer.get_Apply()
        if apply_inner:
            dframe_components = get_min_max_sclr_val(apply_inner,const_out_val,const_out_list,const_in_list,
                                                     const_merged_list,attr_list_one_pp_obj,
                                                     attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif func == "threshold":
        dframe_components = get_binarizer_val(apply_outer,const_out_val, const_out_list, const_in_list,
                                                 const_merged_list, attr_list_one_pp_obj,
                                                 attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list)
    elif func == "sum":
        dframe_components = get_pca_val(apply_outer, const_out_val, const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp, model_list)
    elif func == "product":
        dframe_components = get_poly_feat_val(apply_outer, const_out_val, const_out_list, const_in_list,
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
        dframe_components = get_tfidf_val(apply_outer,const_out_list, const_in_list,
                                             const_merged_list, attr_list_one_pp_obj,
                                             attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif func == "if":
        apply_inner = apply_outer.get_Apply()
        dframe_components = get_imp_val(apply_inner, const_out_val, const_out_list, const_in_list,
                                        const_merged_list, attr_list_one_pp_obj,
                                        attr_list_entire_pp_obj, der_fld_name, der_fld_one_pp_obj, der_filed_entire_pp,
                                        model_list)

    return dframe_components


def get_lbl_binarizer_val(norm_descr, const_out_val, const_out_list, const_in_list,
                          const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                          model_list,const_out_exception):
    const_in_val = None
    field = norm_descr.get_field()
    model = LabelBinarizer()
    dframe_components = store_pp_val(const_out_val, const_out_list, const_in_val, const_in_list, const_merged_list,
                                     field, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                     model_list, model,const_out_exception)
    return dframe_components


def get_norm_discrcomponents(norm_descr, const_out_list, const_in_list, const_merged_list,
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

    dframe_components = get_lbl_binarizer_val(norm_descr, const_out_val, const_out_list, const_in_list,
                          const_merged_list, attr_list_one_pp_obj, attr_list_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                          model_list,const_out_exception)
    return dframe_components



def get_ppcomponents(der_fld,const_out_list,const_in_list,const_merged_list,
                     attr_one_pp_obj,attr_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,
                     model_list,pmml_data_dict):
    dframe_components = list()
    der_fld_name = der_fld.get_name()
    if der_fld.get_Apply():
        apply_outer = der_fld.get_Apply()
        der_fld_name = der_fld.get_name()
        dframe_components = get_applycomponents(apply_outer,const_out_list,const_in_list,
                                                const_merged_list,attr_one_pp_obj,
                                                attr_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)

    elif der_fld.get_NormDiscrete():
        norm_descr = der_fld.get_NormDiscrete()
        der_fld_name = der_fld.get_name()
        dframe_components = get_norm_discrcomponents(norm_descr, const_out_list, const_in_list, const_merged_list,
                                                     attr_one_pp_obj, attr_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                                     model_list,pmml_data_dict)
        return dframe_components

    elif der_fld.get_MapValues():
        map_values = der_fld.get_MapValues()
        der_fld_name = der_fld.get_name()
        dframe_components = get_mapValueComponents(map_values,const_out_list,const_in_list,const_merged_list,
                                                    attr_one_pp_obj,attr_entire_pp_obj,
                                                    der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,model_list)
    elif der_fld.get_TextIndex():
        dframe_components = get_textIndexValueComponent(der_fld,const_out_list,const_in_list,const_merged_list,
                                                            attr_one_pp_obj,attr_entire_pp_obj,der_fld_name,der_fld_one_pp_obj,der_filed_entire_pp,
                                                            model_list)
    return dframe_components


def create_pipeline(**kwargs):
    pipe = None
    if 'model_list' in kwargs.keys():
        if kwargs['model_list']:
            model_list = kwargs['model_list']
            attr_entire_pp_obj = kwargs['attribute_list']
            final_stored_list = list()
            for df_feature_idx in range(len(model_list)):
                main_tuple = (attr_entire_pp_obj[df_feature_idx], model_list[df_feature_idx])
                final_stored_list.append(main_tuple)

            pipe = Pipeline([
                ("mapper", DataFrameMapper([item for item in final_stored_list])),
            ])
    elif 'model' in kwargs.keys():
        model = kwargs['model']
        pipe = Pipeline([
            ("model", model)
        ])
            # pipe = make_pipeline(model)
    return pipe


def get_listed_pp_objs(list_any):
    entire_pp_objs = []
    for item in list_any:
        entire_pp_objs.append([item])
    return entire_pp_objs


def combine_attributes(attr_entire_pp_obj):
    if attr_entire_pp_obj:
        if isinstance(attr_entire_pp_obj[-1],list):
            attr_list = list()
            for attr in attr_entire_pp_obj:
                for single_attr in attr:
                    if single_attr not in attr_list:
                        attr_list.append(single_attr)
            return attr_list
    return attr_entire_pp_obj


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


def get_dtype(dtype):
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
    if 'double' in dtype:
        return np.float64
    if 'integer' in dtype:
        return np.int64
    if 'string' in dtype:
        return str


def store_pp_obj_attributes(const_merged_list, pp_const_merged_list,pipe, data_dict):
    pp_step_count = 0
    for step_num,step in enumerate(pipe.steps):
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
                    dtype = get_dtype(data_dict[attribute])
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
                outer = con_item[0]
                inner = con_item[1]
                outer = np.asarray(outer).astype(np.float64)
                inner = np.asarray(inner).astype(np.float64)
            elif len(con_item) == 1:
                outer = con_item[0]
                outer = np.asarray(outer).astype(np.float64)
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


def store_constants(const_out_list,const_in_list,const_merged_list):
    const_one_derivd_fld_vals = list()
    if const_out_list:
        const_one_derivd_fld_vals.append(const_out_list)
    if const_in_list:
        const_one_derivd_fld_vals.append(const_in_list)
    const_merged_list.append(const_one_derivd_fld_vals)
    return const_merged_list


def set_imputer_strategy(min_strategy):
    if min_strategy == "asMean":
        mining_strategy = "mean"
    elif min_strategy == "asMedian":
        mining_strategy = "median"
    elif min_strategy == "asMode":
        mining_strategy = "most_frequent"
    else:
        mining_strategy = ""
    return mining_strategy


def is_matching_list(list1,list2):
    for list_item in list2:
        if list_item not in list1:
            return False
    return True


def get_imputer_vals(mining_flds,original_attr_entire_pp_obj,original_model_list):
    const_out_list= list()
    const_merged_list = list()
    attr_list_entire_pp_obj = list()
    model_list = list()
    attr_one_pp_obj = list()
    imputer_pp_components = list()
    for model_idx in range(len(original_model_list)):
        if isinstance(original_model_list[model_idx],list):
            if original_model_list[model_idx][0].__class__.__name__ == "Imputer":
                for min_fld in mining_flds:
                    const_out_val = min_fld.get_missingValueReplacement()
                    field = min_fld.get_name()
                    if field in original_attr_entire_pp_obj[model_idx]:
                        attr_one_pp_obj.append(field)
                        const_out_list.append(const_out_val)
                if is_matching_list(original_attr_entire_pp_obj[model_idx],attr_one_pp_obj):
                    model = original_model_list[model_idx][0]
                    const_one_imp__lists = list()
                    model_list.append(model)
                    attr_list_entire_pp_obj.append(attr_one_pp_obj)
                    const_one_imp__lists.append(const_out_list)
                    const_merged_list.append(const_one_imp__lists)
                    attr_one_pp_obj = list()
                    const_out_list = list()
    if model_list:
        model_list = get_listed_pp_objs(model_list)
        imputer_pp_components = const_merged_list,attr_list_entire_pp_obj,model_list
    return imputer_pp_components


def get_original_pp_values(pipeline_obj):
    df_model_list = list()
    attr_entire_pp_obj = list()
    original_pipeline_model_list = list()
    pipe_len = len(pipeline_obj.steps)-1
    for pipe_idx in range(pipe_len):
        single_pipe_obj = pipeline_obj.steps[pipe_idx][1]
        if single_pipe_obj.__class__.__name__ == "DataFrameMapper":
            dframe_mapper_features = single_pipe_obj.features
            dframe_mapper_features_len = len(dframe_mapper_features)
            for dframe_feat_idx in range(dframe_mapper_features_len):
                if not(isinstance(dframe_mapper_features[dframe_feat_idx][1],list)):
                    my_list = list()
                    my_list.append(dframe_mapper_features[dframe_feat_idx][1])
                    df_model_list.append(my_list)
                else:
                    df_model_list.append(dframe_mapper_features[dframe_feat_idx][1])

                if not(isinstance(dframe_mapper_features[dframe_feat_idx][0],list)):
                    my_list = list()
                    my_list.append(dframe_mapper_features[dframe_feat_idx][0])
                    attr_entire_pp_obj.append(my_list)
                else:
                    attr_entire_pp_obj.append(dframe_mapper_features[dframe_feat_idx][0])
        else:
            original_pipeline_model_list.append([single_pipe_obj])


    return attr_entire_pp_obj,df_model_list,original_pipeline_model_list



def is_exception_model(sk_model_obj):
    exception_model = ['LinearRegression','LogisticRegression','SVR','SVC']
    for model_name in exception_model:
        if sk_model_obj.__class__.__name__ == model_name:
            return True
    return False


def arrange_pp_components(sk_model_obj,const_merged_list, fields,attr_entire_pp_obj, model_list,
                                          original_attr_entire_pp_obj, original_df_model_list
                                          ):
    new_const_meregd_list = list()
    new_der_fld_entire_list = list()

    for model_idx, model in enumerate(original_df_model_list):
        for pmml_model_idx, pmml_model_list in enumerate(model_list):
            if len(model) == 1:
                if pmml_model_list[0].__class__.__name__ == model[0].__class__.__name__:
                    if is_matching_list(original_attr_entire_pp_obj[model_idx], attr_entire_pp_obj[pmml_model_idx]):
                        new_const_meregd_list.append(const_merged_list[pmml_model_idx])
                        if not is_exception_model(sk_model_obj):
                            new_der_fld_entire_list.append(fields[pmml_model_idx])

            elif len(model) > 1:
                models_tuple_len = len(model)
                combined_pmml_model = model_list[pmml_model_idx:pmml_model_idx+models_tuple_len]
                combined_pmml_model = combine_attributes(combined_pmml_model)
                if is_nyoka_obj_equal(combined_pmml_model,model):
                    if is_matching_list(original_attr_entire_pp_obj[model_idx], attr_entire_pp_obj[pmml_model_idx]):
                        for idx in range(models_tuple_len):
                            new_const_meregd_list.append(const_merged_list[pmml_model_idx+idx])
                        if not is_exception_model(sk_model_obj):
                            new_der_fld_entire_list.append(fields[pmml_model_idx+idx])

    return new_const_meregd_list, new_der_fld_entire_list

def is_nyoka_obj_equal(obj1,obj2):
    if isinstance(obj1,list):
        if isinstance(obj2,list):
            for first_obj,second_obj in zip(obj1,obj2):
                if first_obj.__class__.__name__ == second_obj.__class__.__name__:
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



def get_wrapped_list(categoric_val_list):
    categoric_val_wrap = list()
    for item in categoric_val_list:
        categoric_val_wrap.append([item])
    return categoric_val_wrap

def get_predictor_val(categoric_fld_parent, pmml_data_dict):
    predictors = categoric_fld_parent.get_CategoricalPredictor()
    categoric_val_one_attr = list()
    categoric_attribute_list = list()
    categoric_val_list = list()
    categoric_model = LabelBinarizer()
    lbl_binarizer_pp_components = tuple()
    const_out_exception = list()
    model_list = list()
    for item in predictors:
        categoric_attribute = item.get_name()
        categoric_val = item.get_value()
        if categoric_attribute not in categoric_attribute_list:
            for data_fld in pmml_data_dict.get_DataField():
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


        # categoric_val_one_attr =

    categoric_val_list.append(categoric_val_one_attr)
    categoric_val_list = get_wrapped_list(categoric_val_list)
    if model_list:
        model_list = get_listed_pp_objs(model_list)
        categoric_attribute_list = get_listed_pp_objs(categoric_attribute_list)
        lbl_binarizer_pp_components = categoric_val_list, categoric_attribute_list, model_list

    return lbl_binarizer_pp_components


def get_lbl_binarizer_exception_val(pmml_modelobj, sk_model_obj,pmml_data_dict):
    lbl_binarizer_pp_components = list()
    if is_exception_model(sk_model_obj):
        if sk_model_obj.__class__.__name__ in ("LinearRegression","LogisticRegression"):
            categoric_fld_parent = pmml_modelobj.get_RegressionTable()[0]
        elif sk_model_obj.__class__.__name__ in ("SVR","SVC"):
            vector_dict = pmml_modelobj.get_VectorDictionary()
            categoric_fld_parent = vector_dict.get_VectorFields()
        lbl_binarizer_pp_components = get_predictor_val(categoric_fld_parent,pmml_data_dict)

    return lbl_binarizer_pp_components


# def combine_list(lbl_binarizer_list, imputer_list):
#     exception_list = list()
#     if lbl_binarizer_list and not imputer_list:
#         exception_list.append(lbl_binarizer_list)
#     elif not lbl_binarizer_list and imputer_list:
#         exception_list.append(imputer_list)
#     elif lbl_binarizer_list and imputer_list:
#         exception_list.append(lbl_binarizer_list)
#         exception_list.append(imputer_list)
#     return exception_list
def segregate_parsed_items(*args):
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


def combine_list(lbl_binarizer_list, imputer_list):
    exception_list = list()
    if lbl_binarizer_list and not imputer_list:
        exception_list.append(lbl_binarizer_list)
    elif not lbl_binarizer_list and imputer_list:
        exception_list.append(imputer_list)
    elif lbl_binarizer_list and imputer_list:
        exception_list.append(lbl_binarizer_list)
        exception_list.append(imputer_list)
    return exception_list


def get_dframe_dtypes(attr_list, pmml_data_dict):
    dtype_dict = dict()
    data_fields = pmml_data_dict.get_DataField()
    for data_fld in data_fields:
        name = data_fld.get_name()
        if name in attr_list:
            dtype = data_fld.get_dataType()
            dtype_dict[name] = dtype

    return dtype_dict


# def sort_dtype(attr_list, data_dict):
#     dframe_data_type = list()
#     for attribute in attr_list:
#         print(attribute)
#         print(data_dict.keys())
#         if attribute in data_dict.keys():
#             dframe_data_type.append(data_dict[attribute])
#     return dframe_data_type


# def get_unique_attribute(original_attr_entire_pp_obj):
#     unique_attr_list = list()
#     for single_attr_list in original_attr_entire_pp_obj:
#         unique_attr_list.append(single_attr_list[0])
#     return unique_attr_list


# def update_datadict(attr_list, unique_attribute, data_dict):
#     for attribute in attr_list:
#         if attribute not in unique_attribute:
#             if attribute in data_dict.keys():
#                 data_dict.pop(attribute)
#
#     return data_dict
def filter_pipe(pipeline_obj):
    pipe = pipeline_obj.steps[:-1]
    if pipe:
        pipe = Pipeline(steps=pipe)
    return pipe


def get_pmml_attributes(pmml_miningobj):
    attribute = list()
    mining_field = pmml_miningobj.get_MiningField()
    for min_fld in mining_field:
        if min_fld.get_usageType() != 'target':
            attribute.append(min_fld.get_name())
    return attribute


def is_ensemble_based(sk_model):
    ensemble_based = ['BaseGradientBoosting','LGBMModel', 'XGBModel','ForestRegressor','ForestClassifier']
    if sk_model.__class__.__base__.__name__ in ensemble_based:
        return True
    else:
        return False


def is_tree_based(sk_model):
    tree_based = ['BaseDecisionTree']
    if sk_model.__class__.__base__.__name__ in tree_based :
        return True
    else:
        return False

def get_pipelineobj(sk_model_obj, der_field, pipeline_obj, pmml_modelobj, pmml):
    const_out_list = list()
    const_in_list = list()
    const_merged_list = list()
    attr_one_pp_obj = list()
    attr_entire_pp_obj = list()
    model_list = list()
    der_fld_one_pp_obj = list()
    der_filed_entire_pp = list()
    fields = list()
    der_fld_combined_data = list()
    pp_const_merged_list = None
    original_pp_components = get_original_pp_values(pipeline_obj)
    original_attr_entire_pp_obj = original_pp_components[0]
    original_df_model_list = original_pp_components[1]
    original_pipeline_model_list = original_pp_components[-1]
    pmml_miningobj = pmml_modelobj.get_MiningSchema()
    pmml_data_dict = pmml.get_DataDictionary()
    mining_flds = pmml_miningobj.get_MiningField()
    lbl_binarizer_list = get_lbl_binarizer_exception_val(pmml_modelobj,sk_model_obj,pmml_data_dict)
    #print(lbl_binarizer_list)


    imputer_list = get_imputer_vals(mining_flds, original_attr_entire_pp_obj, original_df_model_list)
    for der_fld_idx in range(len(der_field)):

        dframe_components = get_ppcomponents(der_field[der_fld_idx], const_out_list, const_in_list, const_merged_list,
                                             attr_one_pp_obj, attr_entire_pp_obj,der_fld_one_pp_obj,der_filed_entire_pp,
                                             model_list,pmml_data_dict)
        const_out_list = dframe_components[0]
        const_in_list = dframe_components[1]
        const_merged_list = dframe_components[2]
        attr_one_pp_obj = dframe_components[3]
        attr_entire_pp_obj = dframe_components[4]
        der_fld_one_pp_obj = dframe_components[5]
        der_filed_entire_pp = dframe_components[6]
        model_list = dframe_components[7]

    if model_list:
        const_merged_list = store_constants(const_out_list, const_in_list, const_merged_list)
        if attr_one_pp_obj:
            attr_entire_pp_obj.append(attr_one_pp_obj)
        model_list = get_listed_pp_objs(model_list)
        der_filed_entire_pp.append(der_fld_one_pp_obj)
        der_fld_combined_data = const_merged_list, attr_entire_pp_obj, model_list, der_filed_entire_pp


    pmml_extracted_data = segregate_parsed_items(lbl_binarizer_list, imputer_list, der_fld_combined_data)
    const_merged_list = pmml_extracted_data[0]
    attr_entire_pp_obj = pmml_extracted_data[1]
    model_list = pmml_extracted_data[2]
    fields = pmml_extracted_data[3]

    original_df_model_list_len = len(original_df_model_list)
    const_merged_list_len = len(const_merged_list)

    if original_pipeline_model_list:
        if const_merged_list_len > original_df_model_list_len:
            if const_merged_list_len > len(original_pipeline_model_list):
                pp_const_merged_list = const_merged_list[const_merged_list_len-len(original_pipeline_model_list):]
                const_merged_list = const_merged_list[:const_merged_list_len-len(original_pipeline_model_list)]
            elif const_merged_list_len == len(original_pipeline_model_list):
                pp_const_merged_list = const_merged_list


    arranged_data = arrange_pp_components(sk_model_obj, const_merged_list, fields, attr_entire_pp_obj, model_list,
                                              original_attr_entire_pp_obj, original_df_model_list
                                          )



    const_merged_list = arranged_data[0]
    if not original_pipeline_model_list:
        fields = arranged_data[1]
    else:
        sub_len = len(original_pipeline_model_list)
        fields = fields[-1]

    if not original_attr_entire_pp_obj:
        original_attr_entire_pp_obj = get_pmml_attributes(pmml_miningobj)
    attr_list = combine_attributes(original_attr_entire_pp_obj)
    fields = combine_attributes(fields)

    # pipe = create_pipeline(attribute_list=original_attr_entire_pp_obj, model_list=original_df_model_list)
    pipe = filter_pipe(pipeline_obj)
    if pipe:
        data_dict = get_dframe_dtypes(attr_list, pmml_data_dict)
        dummy_dframe = create_dummy_dframe(attr_list, data_dict)
        pipe = pipe.fit(dummy_dframe)
        pipe = store_pp_obj_attributes(const_merged_list, pp_const_merged_list, pipe, data_dict)
        if is_tree_based(sk_model_obj) :
            sk_model_obj = get_tree_model(pmml, fields)
        elif is_ensemble_based(sk_model_obj):
            sk_model_obj = get_ensemble_model(pmml, fields)
        pipe.steps.append(("model", sk_model_obj))
    else:
        pipe = create_pipeline(model = sk_model_obj)
    return pipe
