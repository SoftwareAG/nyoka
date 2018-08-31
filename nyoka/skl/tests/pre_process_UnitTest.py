from __future__ import absolute_import
import sys
import os
from pprint import pprint
sys.path.append(os.path.dirname(__file__))
import unittest
import nyoka.PMML43Ext as pml
import nyoka.skl.pre_process as pp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


class TestMethods(unittest.TestCase):

    def test_std_scaler(self):

        trfm_obj = StandardScaler()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)

        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.get_derived_colnames('std_scaler',['displacement']),
            ['std_scaler(displacement)'])


        self.assertEqual(
            pp.std_scaler(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.std_scaler(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "continuous")

        self.assertEqual(
            pp.std_scaler(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")

        self.assertEqual(
            pp.std_scaler(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Constant()[0].get_valueOf_(),
            "{:.25f}".format(trfm_obj.scale_[0]))

        self.assertEqual(
            pp.std_scaler(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Apply()[0].get_Constant()[0].get_valueOf_(),
            "{:.25f}".format(trfm_obj.mean_[0]))

    def test_min_max_scaler(self):

        trfm_obj = MinMaxScaler()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)

        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.get_derived_colnames('min_max_scaler',['displacement']),
            ['min_max_scaler(displacement)'])

        self.assertEqual(
            pp.min_max_scaler(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.min_max_scaler(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "continuous")

        self.assertEqual(
            pp.min_max_scaler(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")

        self.assertEqual(
            pp.min_max_scaler(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Constant()[0].get_valueOf_(),
            "{:.25f}".format(trfm_obj.min_[0]))

        self.assertEqual(
            pp.min_max_scaler(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Apply()[0].get_Constant()[0].get_valueOf_(),
            "{:.25f}".format(trfm_obj.scale_[0]))



    def test_rbst_scaler(self):

        trfm_obj = RobustScaler()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)

        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.get_derived_colnames('rbst_scaler',['displacement']),
            ['rbst_scaler(displacement)'])


        self.assertEqual(
            pp.rbst_scaler(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.rbst_scaler(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "continuous")

        self.assertEqual(
            pp.rbst_scaler(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")

        self.assertEqual(
            pp.rbst_scaler(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Constant()[0].get_valueOf_(),
            "{:.25f}".format(trfm_obj.scale_[0]))

        self.assertEqual(
            pp.rbst_scaler(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Apply()[0].get_Constant()[0].get_valueOf_(),
            "{:.25f}".format(trfm_obj.center_[0]))

    def test_max_abs_scaler(self):

        trfm_obj = MaxAbsScaler()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)

        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.get_derived_colnames('max_abs__scaler', ['displacement']),
            ['max_abs__scaler(displacement)'])

        self.assertEqual(
            pp.max_abs_scaler(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.max_abs_scaler(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "continuous")

        self.assertEqual(
            pp.max_abs_scaler(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")

        self.assertEqual(
            pp.max_abs_scaler(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Constant()[0].get_valueOf_(),
            "{:.25f}".format(trfm_obj.max_abs_[0]))

    def test_binarizer(self):

        trfm_obj = Binarizer()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)

        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.get_derived_colnames('binarizer', ['displacement']),
            ['binarizer(displacement)'])

        self.assertEqual(
            pp.binarizer(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.binarizer(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "continuous")

        self.assertEqual(
            pp.binarizer(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")

        self.assertEqual(
            pp.binarizer(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_Constant()[0].get_valueOf_(),
            trfm_obj.threshold)

    def test_lbl_encoder(self):

        trfm_obj = LabelEncoder()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)
        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.get_derived_colnames('labelEncoder', ['origin']),
            ['labelEncoder(origin)'])

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "continuous")

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")
        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_col_names'][0],
            "labelEncoder(origin)")
        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['pp_feat_class_lbl'][0],
            trfm_obj.classes_[0])

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].get_MapValues().get_outputColumn(),
            "output")

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['pp_feat_name'],
            "origin")

    def test_lbl_encoder(self):

        trfm_obj = LabelEncoder()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)
        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.get_derived_colnames('labelEncoder', ['origin']),
            ['labelEncoder(origin)'])

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "continuous")

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")
        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_col_names'][0],
            "labelEncoder(origin)")
        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['pp_feat_class_lbl'][0],
            trfm_obj.classes_[0])

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['der_fld'][0].get_MapValues().get_outputColumn(),
            "output")

        self.assertEqual(
            pp.lbl_encoder(trfm_obj, feature_names)['pp_feat_name'],
            "origin")

    def test_lbl_binarizer(self):

        trfm_obj = LabelBinarizer()
        trfm_obj, feature_names, target_name = auto_dataset_for_regression(trfm_obj)
        self.assertEqual(
            pp.get_class_name(trfm_obj),
            trfm_obj.__class__.__name__)

        self.assertEqual(
            pp.lbl_binarizer(trfm_obj, feature_names)['der_fld'][0].__class__.__name__,
            pml.DerivedField().__class__.__name__)

        self.assertEqual(
            pp.lbl_binarizer(trfm_obj, feature_names)['der_fld'][0].get_optype(),
            "categorical")

        self.assertEqual(
            pp.lbl_binarizer(trfm_obj, feature_names)['der_fld'][0].get_dataType(),
            "double")

        self.assertEqual(
            pp.lbl_binarizer(trfm_obj, feature_names)['pp_feat_class_lbl'][0],
            trfm_obj.classes_[0])

        self.assertEqual(
            pp.lbl_binarizer(trfm_obj, feature_names)['der_fld'][0].get_NormDiscrete().get_field(),
            "origin")

        self.assertEqual(
            pp.lbl_binarizer(trfm_obj, feature_names)['pp_feat_name'],
            "origin")

    def test_tfidf_vectorizer(self):
        trfm_obj = TfidfVectorizer(norm=None)
        trfm_obj, feature_names, target_name = auto_dataset_for_tfidf_and_count_vec(trfm_obj)
        self.assertEqual(pp.get_class_name(trfm_obj),trfm_obj.__class__.__name__)
        self.assertEqual(
            len(pp.tfidf_vectorizer(trfm_obj, feature_names)['der_col_names']),
            len(trfm_obj.get_feature_names())
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_col_names'][0],
            'tfidf@['+feature_names[0]+']('+trfm_obj.get_feature_names()[0]+')'
        )
        self.assertEqual(
            len(pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'])-1,
            len(trfm_obj.idf_)
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_function(),
            'lowercase'
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['pp_feat_name'],
            feature_names[0]
        )
        self.assertEqual(
            len(pp.tfidf_vectorizer(trfm_obj, feature_names)['pp_feat_class_lbl']), 0
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][0].get_optype(),'categorical'
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][1].get_optype(), 'continuous'
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][0].get_dataType(), 'string'
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][1].get_dataType(), 'double'
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][1].get_Apply().get_Constant()[0].get_valueOf_(),
            trfm_obj.idf_[0]
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][1]
            .get_Apply().get_TextIndex()[0].get_Constant().get_valueOf_(),
            trfm_obj.get_feature_names()[0]
        )
        self.assertEqual(
            pp.tfidf_vectorizer(trfm_obj, feature_names)['der_fld'][-1]
                .get_Apply().get_TextIndex()[0].get_Constant().get_valueOf_(),
            trfm_obj.get_feature_names()[-1]
        )


    def test_count_vectorizer(self):
        trfm_obj = CountVectorizer()
        trfm_obj, feature_names, target_name = auto_dataset_for_tfidf_and_count_vec(trfm_obj)
        self.assertEqual(pp.get_class_name(trfm_obj),trfm_obj.__class__.__name__)
        self.assertEqual(
            len(pp.count_vectorizer(trfm_obj, feature_names)['der_col_names']),
            len(trfm_obj.get_feature_names())
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_col_names'][0],
            'count_vec@['+feature_names[0]+']('+trfm_obj.get_feature_names()[0]+')'
        )
        self.assertEqual(
            len(pp.count_vectorizer(trfm_obj, feature_names)['der_fld'])-1,
            len(trfm_obj.get_feature_names())
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_fld'][0].get_Apply().get_function(),
            'lowercase'
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['pp_feat_name'],
            feature_names[0]
        )
        self.assertEqual(
            len(pp.count_vectorizer(trfm_obj, feature_names)['pp_feat_class_lbl']), 0
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_fld'][0].get_optype(),'categorical'
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_fld'][1].get_optype(), 'continuous'
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_fld'][0].get_dataType(), 'string'
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_fld'][1].get_dataType(), 'double'
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_fld'][1]
            .get_TextIndex().get_Constant().get_valueOf_(),
            trfm_obj.get_feature_names()[0]
        )
        self.assertEqual(
            pp.count_vectorizer(trfm_obj, feature_names)['der_fld'][-1]
                .get_TextIndex().get_Constant().get_valueOf_(),
            trfm_obj.get_feature_names()[-1]
        )


def iris_dataset_for_classification(sk_model):
    df = pd.read_csv('iris.csv')
    feature_names = df.columns.drop('species')
    feature_names = feature_names._data
    target_name = 'species'
    x_train, x_test, y_train, y_test = train_test_split(df[feature_names], df[target_name], test_size=0.33,
                                                        random_state=101)
    sk_model=sk_model.fit(x_train,y_train)
    return sk_model,feature_names,target_name


def auto_dataset_for_regression(trfm_obj):
    df = pd.read_csv('auto-mpg.csv')
    X = df.drop(['mpg','car name'], axis=1)
    y = df['mpg']
    feature_names = [name for name in df.columns if name not in ('mpg','car name')]
    target_name='mpg'
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=101)
    if "LabelEncoder" or "LabelBinarizer" in type(trfm_obj):
        x_train=np.array(x_train['origin']).reshape(-1,1)
        sk_model=trfm_obj.fit(x_train)
        feature_names = ['origin']
    else:
        sk_model = trfm_obj.fit(x_train)
    return sk_model,feature_names,target_name

def auto_dataset_for_tfidf_and_count_vec(trfm_obj):
    df = pd.read_csv('auto-mpg.csv')
    X = df.drop(['mpg'], axis=1)
    y = df['mpg']
    feature_names = [name for name in df.columns if name not in ('mpg')]
    target_name = 'mpg'
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    sk_model = trfm_obj.fit(x_train)
    return sk_model, feature_names, target_name


if __name__=='__main__':
    unittest.main(warnings='ignore')







