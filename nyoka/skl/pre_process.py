from __future__ import absolute_import
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import PMML43Ext as pml
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))
exception_cols = list()

def get_preprocess_val(ppln_sans_predictor, initial_colnames, model, model_name):
    """

    Parameters
    ----------
    model :
        Contains an instance of Sklearn model
    ppln_sans_predictor :
        Contains an instance of Sklearn Pipeline
    initial_colnames : list
        Contains list of feature/column names.

    Returns
    -------
    pml_pp: dictionary
    Returns a dictionary that contains data related to pre-processing

    """
    pml_pp = dict()
    pml_derived_flds = list()
    initial_colnames = [col_name for col_name in initial_colnames]
    updated_colnames = initial_colnames.copy()
    dtd_feat_names = list()
    classes = list()
    class_attribute = list()
    mining_strategy = list()
    mining_replacement_val = list()
    mining_attributes = list()
    derived_flds_hidden = list()
    pml_trfm_dict = list()
    polynomial_features.poly_ctr = 0
    pca.counter = 0
    imputer.col_names = initial_colnames

    for ppln_step in ppln_sans_predictor:
        ppln_step_inst = ppln_step[1]
        if "DataFrameMapper" == get_class_name(ppln_step_inst):
            dfm_steps = ppln_step_inst.features
            dfm_col_names = list()
            for dfm_step in dfm_steps:
                dfm_step_col_names = dfm_step[0]
                dfm_step_trfms = dfm_step[1]
                if not dfm_step_trfms:
                    for col in dfm_step_col_names:
                        if col not in dtd_feat_names:
                            dtd_feat_names.append(col)
                    for col in dfm_step_col_names:
                        if col not in dfm_col_names:
                            dfm_col_names.append(col)
                    continue
                if not hasattr(dfm_step_col_names, "__len__") or isinstance(dfm_step_col_names, str):
                    dfm_step_col_names = [dfm_step_col_names]
                if not hasattr(dfm_step_trfms, "__len__") or isinstance(dfm_step_trfms, str):
                    dfm_step_trfms = [dfm_step_trfms]
                for name in dfm_step_col_names:
                    if name not in dtd_feat_names:
                        dtd_feat_names.append(name)

                for trfm in dfm_step_trfms:
                    pp_dict = get_pml_derived_flds(trfm, dfm_step_col_names, derived_fld=derived_flds_hidden,
                                                   model=model)
                    derived_flds = pp_dict['der_fld']
                    derived_names = pp_dict['der_col_names']
                    if 'pp_feat_class_lbl' in pp_dict.keys():
                        classes.append(pp_dict['pp_feat_class_lbl'])
                        class_attribute.append(pp_dict['pp_feat_name'])
                    if 'pp_feat_class_ohe' in pp_dict.keys():
                        classes.append(pp_dict['pp_feat_class_ohe'])
                        class_attribute.append(pp_dict['pp_feat_name'])
                    if 'mining_strategy' in pp_dict.keys():
                        mining_attributes.append(pp_dict['der_col_names'])
                        mining_strategy.append(pp_dict['mining_strategy'])
                        mining_replacement_val.append(pp_dict['mining_replacement_val'])
                    if 'hidden_lb_der_flds' in pp_dict.keys():
                        derived_flds_hidden.extend(pp_dict['hidden_lb_der_flds'])
                    if 'hidden_ohe_der_flds' in pp_dict.keys():
                        derived_flds_hidden.extend(pp_dict['hidden_ohe_der_flds'])
                    pml_derived_flds.extend(derived_flds)
                    dfm_step_col_names = derived_names
                dfm_col_names.extend(derived_names)

            updated_colnames = dfm_col_names
        else:
            if not dtd_feat_names:
                dtd_feat_names = initial_colnames
                updated_colnames = initial_colnames
            if not hasattr(ppln_step_inst, "__len__") or isinstance(ppln_step_inst, str):
                ppln_step_inst = [ppln_step_inst]
            for trfm in ppln_step_inst:
                pp_dict = get_pml_derived_flds(trfm, updated_colnames, derived_fld=derived_flds_hidden, model=model)
                derived_flds = pp_dict['der_fld']
                derived_names = pp_dict['der_col_names']
                if 'pp_feat_class_lbl' in pp_dict.keys():
                    classes.append(pp_dict['pp_feat_class_lbl'])
                    class_attribute.append(pp_dict['pp_feat_name'])
                if 'pp_feat_class_ohe' in pp_dict.keys():
                    classes.append(pp_dict['pp_feat_class_ohe'])
                    class_attribute.append(pp_dict['pp_feat_name'])
                if 'mining_strategy' in pp_dict.keys():
                    mining_attributes.append(pp_dict['der_col_names'])
                    mining_strategy.append(pp_dict['mining_strategy'])
                    mining_replacement_val.append(pp_dict['mining_replacement_val'])
                pml_derived_flds.extend(derived_flds)
                updated_colnames = derived_names

    if pml_derived_flds:
        pml_trfm_dict = pml.TransformationDictionary(DerivedField=pml_derived_flds,for_=model_name)
    pml_pp['trfm_dict'] = pml_trfm_dict
    pml_pp['derived_col_names'] = updated_colnames
    pml_pp['preprocessed_col_names'] = dtd_feat_names
    pml_pp['categorical_feat_values'] = classes, class_attribute
    pml_pp['mining_imp_values'] = mining_attributes, mining_strategy, mining_replacement_val

    return pml_pp


def get_class_name(cls):
    """
    Parameters
    ----------
    cls :
        Contains the Sklearn's preprocessing instance

    Returns
    -------
    cls.__class__.__name__: String
        Returns the class name of the pre-processed object.

    """
    return cls.__class__.__name__


def get_pml_derived_flds(trfm, col_names, **kwargs):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pml_pp: dictionary
        Returns a dictionary that contains attributes related to any preprocessing function .

    """

    if "StandardScaler" == get_class_name(trfm):
        return std_scaler(trfm, col_names, **kwargs)
    elif "MinMaxScaler" == get_class_name(trfm):
        return min_max_scaler(trfm, col_names)
    elif "RobustScaler" == get_class_name(trfm):
        return rbst_scaler(trfm, col_names)
    elif "MaxAbsScaler" == get_class_name(trfm):
        return max_abs_scaler(trfm, col_names)
    elif "TfidfVectorizer" == get_class_name(trfm):
        return tfidf_vectorizer(trfm, col_names)
    elif "CountVectorizer" == get_class_name(trfm):
        return count_vectorizer(trfm, col_names)
    elif "LabelEncoder" == get_class_name(trfm):
        return lbl_encoder(trfm, col_names)
    elif "Imputer" == get_class_name(trfm):
        return imputer(trfm, col_names, **kwargs)
    elif "Binarizer" == get_class_name(trfm):
        return binarizer(trfm, col_names)
    elif "PolynomialFeatures" == get_class_name(trfm):
        return polynomial_features(trfm, col_names)
    elif "PCA" == get_class_name(trfm):
        return pca(trfm, col_names)
    elif "LabelBinarizer" == get_class_name(trfm):
        return lbl_binarizer(trfm, col_names, **kwargs)
    elif "OneHotEncoder"==get_class_name(trfm):
        return one_hot_encoder(trfm,col_names,**kwargs)
    elif "CategoricalImputer" == get_class_name(trfm):
        return cat_imputer(trfm, col_names)
    else:
        raise TypeError("This PreProcessing Task is not Supported")


def get_derived_colnames(trfm_name, col_names, *args):
    """

    Parameters
    ----------
    trfm_name : String
        Name of the derived field to be assigned after preprocessing
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pml_pp: list
        Returns a list that contains names of the preprocessed features.

    """
    extra_symbol = ""
    if args:
        extra_symbol = args[0]
    derived_colnames = list()
    for col_name in col_names:
        derived_colnames.append(trfm_name + '(' + str(col_name) + ')' + extra_symbol)

    return derived_colnames


def unround_scalers(scalar_val):  # not sure of its purpose ------------------>>>>
    """

    Parameters
    ----------
    scalar_val : float
        A numpy float value

    Returns
    -------
    unround_val: float
        Returns a numpy floating point number with a precision of 16 digits after decimal.

    """
    unround_val = '{:.25f}'.format(scalar_val)
    return unround_val


def any_in(seq_a, seq_b):
    """

    Parameters
    ----------
    seq_a : list
        A list of items

    seq_b : list
        A list of items

    Returns
    -------
    seq_a: bool
        Returns a boolean value if any item of seq_a belongs to seq_b or visa versa

    """
    return any(elem in seq_b for elem in seq_a)


# Methods for Preprocessings


def imputer(trfm, col_names, **kwargs):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's Imputer preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to Imputer preprocessing.

    """
    original_col_names = imputer.col_names
    derived_colnames = col_names
    pp_dict = dict()
    derived_flds = list()

    model = kwargs['model']

    mining_strategy = trfm.strategy
    if "mean" in mining_strategy:
        mining_strategy = "asMean"
    elif "median" in mining_strategy:
        mining_strategy = "asMedian"
    elif "most_frequent" in mining_strategy:
        mining_strategy = "asMode"
    mining_replacement_val = trfm.statistics_

    if not any_in(original_col_names, col_names):
        derived_colnames = get_derived_colnames('imputer', col_names)
        for col_name_idx in range(len(col_names)):
            if (col_names[col_name_idx] not in exception_cols):
                const_list = list()
                apply_inner = list()
                apply_inner.append(pml.Apply(function='isMissing', FieldRef=[pml.FieldRef(field=col_names[col_name_idx])]))
                const_obj = pml.Constant(
                    dataType="double",  # <---------------------
                    valueOf_=mining_replacement_val[col_name_idx]
                ),
                fieldref_obj = pml.FieldRef(field=col_names[col_name_idx])
                fieldref_obj.original_tagname_ = "FieldRef"
                const_list.append(const_obj[0])
                const_list.append(fieldref_obj)
                apply_outer = pml.Apply(
                    Apply_member=apply_inner,
                    function='if',
                    Constant=const_list
                )

                derived_flds.append(pml.DerivedField(
                    Apply=apply_outer,
                    name=derived_colnames[col_name_idx],
                    optype="continuous",
                    dataType="double"
                ))
    else:
        pp_dict['mining_strategy'] = mining_strategy
        pp_dict['mining_replacement_val'] = mining_replacement_val
        pp_dict['mining_attributes'] = col_names

    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict


def cat_imputer(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's Imputer preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to Imputer preprocessing.

    """
    derived_colnames = col_names
    pp_dict = dict()
    derived_flds = list()

    mining_strategy = "asMode"
    mining_replacement_val = trfm.fill_

    pp_dict['mining_strategy'] = mining_strategy
    pp_dict['mining_replacement_val'] = mining_replacement_val
    pp_dict['mining_attributes'] = col_names
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames

    return pp_dict


def pca(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's PCA preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to PCA preprocessing.

    """
    pca.counter += 1
    pp_dict = dict()
    derived_flds = list()
    derived_colnames = list()
    val = trfm.mean_
    zero = 0.0
    for preprocess_idx in range(trfm.n_components_):
        add = list()
        for pca_idx in range(trfm.n_features_):
            apply_inner = pml.Apply(function='-',
                                    Constant=[pml.Constant(dataType="double",
                                                           valueOf_="{:.16f}".format(val[pca_idx]))],
                                    FieldRef=[pml.FieldRef(field=col_names[pca_idx])])
            apply_outer = pml.Apply(function="*",
                                    Apply_member=[apply_inner],
                                    Constant=[pml.Constant(dataType="double",
                                                           valueOf_=zero if trfm.components_[preprocess_idx][
                                                                                pca_idx] == 0.0 else
                                                           "{:.16f}".format(trfm.components_[preprocess_idx][pca_idx]))])
            add.append(apply_outer)
        app0 = pml.Apply(function="sum", Apply_member=add)

        derived_flds.append(pml.DerivedField(Apply=app0,
                                             dataType="double",
                                             optype="continuous",
                                             name="PCA" + str(pca.counter) + "-" + str(preprocess_idx)))
        name = derived_flds[preprocess_idx].get_name()
        derived_colnames.append(name)
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict

def remove_punctuation(word):
    # no_punct_word = regex.sub('', word)
    return regex.findall(word)
    # return no_punct_word


def tfidf_vectorizer(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's TfIdfVectorizer preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to TfIdfVectorizer preprocessing.

    """
    pp_dict = dict()
    features = [str(feat.encode("utf8"))[2:-1] for feat in trfm.get_feature_names()]
    # extra_features = [str(feat.encode("utf8"))[2:-1] for feat in list(trfm.vocabulary_.keys())]
    # features = trfm.get_feature_names()
    idfs = trfm.idf_
    # extra_features = list(trfm.vocabulary_.keys())
    derived_flds = list()
    # derived_colnames = list()
    derived_colnames = get_derived_colnames('tfidf@[' + col_names[0] + ']', features)
    if trfm.lowercase:
        derived_flds.append(
            pml.DerivedField(name='lowercase(' + col_names[0] + ')',
                            optype='categorical', dataType='string',
                            Apply=pml.Apply(function='lowercase',
                                            FieldRef=[pml.FieldRef(field=col_names[0])])))
    for feat_idx, idf in zip(range(len(features)), idfs):
        # no_punct_word = remove_punctuation(features[feat_idx])
        # if len(no_punct_word) == 0:
            # df_name = 'tfidf_vec@[' + col_names[0] + ']('+ features[feat_idx]+')'
            # derived_colnames.append(df_name)
        derived_flds.append(pml.DerivedField(
            # name=df_name,
            name = derived_colnames[feat_idx],
            optype='continuous',
            dataType='double',
            Apply=pml.Apply(function='*',
                            TextIndex=[pml.TextIndex(textField='lowercase(' + col_names[0] + ')' if trfm.lowercase \
                                else col_names[0],
                                                    # wordSeparatorCharacterRE='\s+',
                                                    wordSeparatorCharacterRE=trfm.token_pattern,
                                                    tokenize='true',
                                                    Constant=pml.Constant(valueOf_=features[feat_idx]),
                                                    # Extension=[pml.Extension(anytypeobjs_=[extra_features[feat_idx]])]
                                                    )],
                            Constant=[pml.Constant(valueOf_="{:.16f}".format(idf))])
                            ))
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    pp_dict['pp_feat_name'] = col_names[0]
    pp_dict['pp_feat_class_lbl'] = list()
    return pp_dict


def count_vectorizer(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's CountVectorizer preprocessing instance.
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to CountVectorizer preprocessing.

    """
    pp_dict = dict()
    features = [str(feat.encode("utf8"))[2:-1] for feat in trfm.get_feature_names()]
    # extra_features = [str(feat.encode("utf8"))[2:-1] for feat in list(trfm.vocabulary_.keys())]
    # features = trfm.get_feature_names()
    # extra_features = list(trfm.vocabulary_.keys())
    derived_flds = list()
    derived_colnames = get_derived_colnames('count_vec@[' + col_names[0] + ']', features)
    # derived_colnames = list()
    if trfm.lowercase:
        derived_flds.append(pml.DerivedField(name='lowercase(' + col_names[0] + ')',
                                            optype='categorical',
                                            dataType='string',
                                            Apply=pml.Apply(function='lowercase',
                                                            FieldRef=[pml.FieldRef(field=col_names[0])])))
    for imp_features, index in zip(features, range(len(features))):
        # no_punct_word = remove_punctuation(imp_features)
        # if len(no_punct_word) == 0:
            # df_name = 'count_vec@[' + col_names[0] + ']('+ imp_features+')'
            # derived_colnames.append(df_name)
        df_name = derived_colnames[index]
        derived_flds.append(pml.DerivedField(name=df_name,
                                            optype='continuous',
                                            dataType='double',
                                            TextIndex=pml.TextIndex(textField='lowercase(' + col_names[0] + ')' if trfm.lowercase \
                                                else col_names[0],
                                                                    # wordSeparatorCharacterRE='\s+',
                                                                    wordSeparatorCharacterRE=trfm.token_pattern,
                                                                    tokenize='true',
                                                                    Constant=pml.Constant(dataType="string",
                                                                                        valueOf_=imp_features),
                                                                    # Extension=[pml.Extension(
                                                                    #     anytypeobjs_=[extra_features[index]])]
                                                                    )))
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    pp_dict['pp_feat_name'] = col_names[0]
    pp_dict['pp_feat_class_lbl'] = list()
    return pp_dict


def is_present(string1,iterator):
    if isinstance(iterator,(list,tuple)):
        for iterator_item in iterator:
            if string1 in iterator_item:
                return True
    elif isinstance(iterator,str):
        if string1 in iterator:
            return True

    return False


def std_scaler(trfm, col_names, **kwargs):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's Standard Scaler preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to Standard Scaler preprocessing.

    """
    derived_flds = list()
    pp_dict = dict()
    if is_present("labelBinarizer",col_names):
        derived_flds_hidden = kwargs['derived_fld']
        if derived_flds_hidden:
            derived_flds.extend(derived_flds_hidden)

    derived_colnames = get_derived_colnames('standardScaler', col_names)
    for col_name_idx in range(len(col_names)):
        apply_inner = list()
        apply_inner.append(pml.Apply(
            function='-',
            Constant=[pml.Constant(
                dataType="double",  # <---------------------
                valueOf_="{:.16f}".format(trfm.mean_[col_name_idx])
            )],
            FieldRef=[pml.FieldRef(field=col_names[col_name_idx])]
        ))
        apply_outer = pml.Apply(
            Apply_member=apply_inner,
            function='/',
            Constant=[pml.Constant(
                dataType="double",  # <----------------------------
                valueOf_="{:.16f}".format(trfm.scale_[col_name_idx])
            )]
        )
        derived_flds.append(pml.DerivedField(
            Apply=apply_outer,
            name=derived_colnames[col_name_idx],
            optype="continuous",
            dataType="double"
        ))


    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict


def min_max_scaler(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's MinMaxScaler preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to MinMaxScaler preprocessing.

    """
    pp_dict = dict()
    derived_flds = list()
    # col_names = list(filter(lambda x: x not in exception_cols, col_names))
    derived_colnames = get_derived_colnames("minMaxScaler", col_names)
    for col_name_idx in range(len(col_names)):
        if(col_names[col_name_idx] not in exception_cols):
            apply_inner = list()
            apply_inner.append(pml.Apply(
                function='*',
                Constant=[pml.Constant(
                    dataType="double",
                    valueOf_="{:.16f}".format(trfm.scale_[col_name_idx])
                )],
                FieldRef=[pml.FieldRef(field=col_names[col_name_idx])]
            ))
            apply_outer = pml.Apply(
                Apply_member=apply_inner,
                function='+',
                Constant=[pml.Constant(
                    dataType="double",
                    valueOf_="{:.16f}".format(trfm.min_[col_name_idx])
                )]
            )
            derived_flds.append(pml.DerivedField(
                Apply=apply_outer,
                name=derived_colnames[col_name_idx],
                optype="continuous",
                dataType="double"
            ))
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict


def rbst_scaler(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's RobustScaler preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to RobustScaler preprocessing.

    """
    pp_dict = dict()
    derived_flds = list()
    derived_colnames = get_derived_colnames('robustScaler', col_names)
    for col_name_idx in range(len(col_names)):
        if (col_names[col_name_idx] not in exception_cols):
            apply_inner = list()
            apply_inner.append(pml.Apply(
                function='-',
                Constant=[pml.Constant(
                    dataType="double",  # <---------------------
                    valueOf_="{:.16f}".format(trfm.center_[col_name_idx])
                )],
                FieldRef=[pml.FieldRef(field=col_names[col_name_idx])],
                Extension=[pml.Extension(name='scaling', anytypeobjs_=['RobustScaler'])]
            ))
            apply_outer = pml.Apply(
                Apply_member=apply_inner,
                function='/',
                Constant=[pml.Constant(
                    dataType="double",  # <----------------------------
                    valueOf_="{:.16f}".format(trfm.scale_[col_name_idx])
                )]
            )
            derived_flds.append(pml.DerivedField(
                Apply=apply_outer,
                name=derived_colnames[col_name_idx],
                optype="continuous",
                dataType="double"
            ))
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict


def max_abs_scaler(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's MaxabsScaler preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to MaxabsScaler preprocessing.

    """
    pp_dict = dict()
    derived_flds = list()
    derived_colnames = get_derived_colnames('maxAbsScaler', col_names)
    for col_name_idx in range(len(col_names)):
        if (col_names[col_name_idx] not in exception_cols):
            apply_outer = pml.Apply(
                function='/',
                Constant=[pml.Constant(
                    dataType="double",  # <---------------------
                    valueOf_="{:.16f}".format(trfm.max_abs_[col_name_idx])
                )],
                FieldRef=[pml.FieldRef(field=col_names[col_name_idx])]
            )

            derived_flds.append(pml.DerivedField(
                Apply=apply_outer,
                name=derived_colnames[col_name_idx],
                optype="continuous",
                dataType="double"
            ))
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict


def lbl_encoder(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's LabelEncoder preprocessing instance
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to LabelEncoder preprocessing.

    """
    pp_dict = dict()
    derived_flds = list()
    field_column_pair = list()
    rows = []
    categoric_lbls = trfm.classes_.tolist()
    categoric_lbls_num = trfm.transform(trfm.classes_.tolist()).tolist()
    derived_colnames = get_derived_colnames('labelEncoder', col_names)
    for row_idx in range(len(categoric_lbls_num)):
        row_main = pml.row()
        row_main.elementobjs_ = ['input', 'output']
        row_main.input = categoric_lbls[row_idx]
        row_main.output = str(categoric_lbls_num[row_idx])
        rows.append(row_main)
    field_column_pair.append(pml.FieldColumnPair(field=str(col_names[0]), column="input"))
    inline_table = pml.InlineTable(row=rows)
    map_values = pml.MapValues(outputColumn="output", FieldColumnPair=field_column_pair, InlineTable=inline_table)
    derived_flds.append(
        pml.DerivedField(MapValues=map_values, name=derived_colnames[0], optype="continuous", dataType="double"))

    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    pp_dict['pp_feat_class_lbl'] = categoric_lbls
    pp_dict['pp_feat_name'] = col_names[0]

    return pp_dict


def binarizer(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's Binarizer preprocessing instance.
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to Binarizer preprocessing.

    """
    pp_dict = dict()
    derived_flds = list()
    derived_colnames = get_derived_colnames("binarizer", col_names)
    for col_name_idx in range(len(col_names)):
        apply_outer = pml.Apply(
            function='threshold',
            Constant=[pml.Constant(
                dataType="double",
                valueOf_=trfm.threshold
            )],
            FieldRef=[pml.FieldRef(field=col_names[col_name_idx])])

        derived_flds.append(pml.DerivedField(
            Apply=apply_outer,
            name=derived_colnames[col_name_idx],
            optype="continuous",
            dataType="double"
        ))

    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict


def polynomial_features(trfm, col_names):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's PolynomialFeatures preprocessing instance.
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to PolynomialFeatures preprocessing.

    """
    polynomial_features.poly_ctr += 1
    pp_dict = dict()
    derived_flds = []
    derived_colnames = []

    for polyfeat_idx in range(trfm.powers_.shape[0]):
        apply_inner_container = []
        for col_name_idx in range(len(col_names)):
            val = int(trfm.powers_[polyfeat_idx][col_name_idx])
            apply_inner = pml.Apply(
                function='pow',
                Constant=[pml.Constant(
                    dataType="integer",
                    valueOf_=val
                )],
                FieldRef=[pml.FieldRef(field=col_names[col_name_idx])])
            apply_inner_container.append(apply_inner)
        apply_outer = pml.Apply(function="product",
                                Apply_member=apply_inner_container
                                )
        derived_flds.append(pml.DerivedField(
            Apply=apply_outer,
            dataType="double",
            optype="continuous",
            name="poly" + str(polynomial_features.poly_ctr) + '-' + "x" + str(polyfeat_idx)
        ))
        name = derived_flds[polyfeat_idx].get_name()
        derived_colnames.append(name)
    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    return pp_dict


def lbl_binarizer(trfm, col_names, **kwargs):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's Label Binarizer preprocessing instance.
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to Label Binarizer preprocessing.

    """
    derived_flds = list()
    derived_colnames = list()
    pp_dict = dict()
    categoric_lbls = trfm.classes_.tolist()
    model_exception_list = ["LinearRegression", "LogisticRegression", "SVR", "SVC"]
    model = kwargs['model']
    for col_name_idx in range(len(col_names)):
        if len(categoric_lbls) == 2:
            derived_colnames = get_derived_colnames("labelBinarizer(" + str(col_names[col_name_idx]),
                                                    [categoric_lbls[-1]], ")")

            norm_descr = pml.NormDiscrete(field=str(col_names[-1]), value=str(categoric_lbls[-1]))
            derived_flds.append(pml.DerivedField(NormDiscrete=norm_descr,
                                                 name=derived_colnames[-1],
                                                 optype="categorical",
                                                 dataType="double"))
        else:
            derived_colnames = get_derived_colnames("labelBinarizer(" + str(col_names[col_name_idx]),
                                                    categoric_lbls, ")")
            for attribute_name in col_names:
                for class_name, class_idx in zip(categoric_lbls, range(len(categoric_lbls))):
                    norm_descr = pml.NormDiscrete(field=str(attribute_name), value=str(class_name))
                    derived_flds.append(
                        pml.DerivedField(NormDiscrete=norm_descr,
                                         name=derived_colnames[class_idx],
                                         optype="categorical",
                                         dataType="double"))
    if any_in([model.__class__.__name__], model_exception_list):
        pp_dict['hidden_lb_der_flds'] = derived_flds
        exception_cols.extend(derived_colnames)
        derived_flds = list()

    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    pp_dict['pp_feat_class_lbl'] = categoric_lbls
    pp_dict['pp_feat_name'] = col_names[0]

    return pp_dict


def one_hot_encoder(trfm, col_names, **kwargs):
    """

    Parameters
    ----------
    trfm :
        Contains the Sklearn's One hot encoder preprocessing instance.
    col_names : list
        Contains list of feature/column names.
        The column names may represent the names of preprocessed attributes.

    Returns
    -------
    pp_dict : dictionary
        Returns a dictionary that contains attributes related to Label Binarizer preprocessing.

    """
    derived_flds = list()
    derived_colnames = list()
    pp_dict = dict()
    categoric_lbls = trfm.active_features_.tolist()
    model_exception_list = ["LinearRegression", "LogisticRegression", "SVR", "SVC"]
    model = kwargs['model']
    for col_name_idx in range(len(col_names)):
        derived_colnames = get_derived_colnames("oneHotEncoder(" + str(col_names[col_name_idx]),
                                                categoric_lbls, ")")
        for attribute_name in col_names:
            for class_name, class_idx in zip(categoric_lbls, range(len(categoric_lbls))):
                norm_descr = pml.NormDiscrete(field=str(attribute_name), value=str(class_name))
                derived_flds.append(
                    pml.DerivedField(NormDiscrete=norm_descr,
                                     name=derived_colnames[class_idx],
                                     optype="categorical",
                                     dataType="double"))
    if any_in([model.__class__.__name__], model_exception_list):
        pp_dict['hidden_ohe_der_flds'] = derived_flds
        exception_cols.extend(derived_colnames)
        derived_flds = list()

    pp_dict['der_fld'] = derived_flds
    pp_dict['der_col_names'] = derived_colnames
    pp_dict['pp_feat_class_ohe'] = categoric_lbls
    pp_dict['pp_feat_name'] = col_names[0]

    return pp_dict
