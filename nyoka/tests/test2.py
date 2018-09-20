#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code is devloped for the task ZENA-19.
Write a short test program that creates the PMML tags through nyoka.pmml.PMML43,
using the classes and their initializers there.
Picked example published : http://dmg.org/pmml/pmml_examples/knime_pmml_examples/ElNinoLinearReg.xml
Write a set of unit tests that compare the original file with the one exported by nyoka.pmml
"""

# python imports
import sys
import datetime
import numpy
import os

# nyoka import PMML43Ext
from nyoka.PMML43Ext import *


def create_data_dict_with_fields():
    """ creates a DataDictionary with necessary DattFields """

    # create data dictionary
    data_dict = DataDictionary()

    # list of values for datafields
    names = ["latitude","longitude","zon_winds","mer_winds","humidity","airtemp","s_s_temp"]
    left_margins = ["-8.28","-179.97","-8.9","-6.4","27.58","22.72","22.44"]
    right_margins = ["8.97", "179.8", "7.0", "7.1", "99.4", "30.04", "30.34"]

    # loop over and add DataFields to DataDictionary
    for indx, name in enumerate(names):
        interval = Interval(closure="closedClosed",
                            leftMargin=left_margins[indx],
                            rightMargin=right_margins[indx])
        data_field = DataField(name=name, optype='continuous', dataType='double')
        data_field.add_Interval(interval)
        data_dict.add_DataField(data_field)
    return data_dict

def create_resgression_model_with_fields():
    """ creates a RegressionModel with MiningSchema and RegressionTable to generate the example above"""

    # for regression model fields
    names =  ["latitude", "longitude", "zon_winds", "mer_winds", "humidity", "s_s_temp"]
    coeffs=["3.363167396766842E-4", "1.238009786077277E-4", "-0.07364295448649694",
            "-0.04315230485415502", "-0.011583900555823673", "0.7840777698224044"]

    # Mininig Schema
    mining_schema =  MiningSchema()

    # MiningFields
    for indx, name in enumerate(names):
        mining_field = MiningField(name=name, invalidValueTreatment="asIs")
        mining_schema.add_MiningField(mining_field)
    mining_field = MiningField(name="airtemp", usageType="predicted", invalidValueTreatment="asIs")
    mining_schema.add_MiningField(mining_field)

    # Regression Table
    regression_table = RegressionTable(intercept="6.008706171265235")
    for indx, name in enumerate(names):
        numeric_predictor = NumericPredictor(name=name, coefficient=coeffs[indx])
        regression_table.add_NumericPredictor(numeric_predictor)

    # create regression model
    regression_model = RegressionModel(modelName="KNIME Linear Regression",
                                       functionName="regression",
                                       algorithmName="LinearRegression",
                                       targetFieldName="airtemp")
    regression_model.MiningSchema = mining_schema
    regression_model.add_RegressionTable(regression_table)
    return regression_model


def main():
    """ Main test function to create the pmml file """
    timestamp = Timestamp(datetime.datetime.now())
    header = Header(copyright="(C) 2017 PB", Timestamp=timestamp)
    data_dict = create_data_dict_with_fields()
    regression_model = create_resgression_model_with_fields()
    pmml = PMML(Header=header, DataDictionary=data_dict)
    pmml.add_RegressionModel(regression_model)
    if os.path.isdir("nyoka/tests"):
        pmml.export(open("nyoka/tests/regression_model.pmml","w"),0,"")
    elif os.path.isdir("tests"):
        pmml.export(open("tests/regression_model.pmml","w"),0,"")
    mining_schema = regression_model.MiningSchema
    regression_table = regression_model.get_RegressionTable()


    # cross verification test cases by parsing the same file and comparing different attributes
    if os.path.isdir("nyoka/tests"):
        pmml2 = parse("nyoka/tests/regression_model.pmml", False)
    elif os.path.isdir("tests"):
        pmml2 = parse("tests/regression_model.pmml", False)
    header2 = pmml2.Header
    timestamp2 = header2.Timestamp
    regression_model2 = pmml2.RegressionModel[0]
    mining_schema2 = regression_model2.MiningSchema
    resgression_table2 = regression_model2.get_RegressionTable()

    # test cases
    assert dir(pmml2) == dir(pmml)
    assert dir(header2) == dir (header)
    assert dir(timestamp2) == dir(timestamp)
    assert dir(regression_model2) == dir(regression_model)
    assert header2.copyright == header.copyright
    assert timestamp2.valueOf_.strip().replace("\n","") == timestamp.valueOf_.strip().replace("\n","")
    assert dir(mining_schema2) == dir(mining_schema)
    assert dir(resgression_table2) == dir(regression_table)

if __name__ == '__main__':
    main()
