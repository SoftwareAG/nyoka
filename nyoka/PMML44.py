#!/usr/bin/env python
#
# Generated Thu Jun 24 16:39:10 2021 by generateDS.py version 2.28a.
#
# Command line options:
#   ('--no-warnings', '')
#   ('--export', 'write literal etree')
#   ('--super', 'nyoka.PMML44Super')
#   ('--subclass-suffix', '')
#   ('-o', 'nyoka.PMML44Super.py')
#   ('-s', 'nyoka.PMML44.py')
#   ('-b', 'behaviorsDir.xml')
#   ('-f', '')
#
# Command line arguments:
#   ..\nyoka.PMML44.xsd
#
# Command line:
#   C:\Users\NIBO\OneDrive - Software AG\projects\New folder\nyoka\nyoka\PMML44\gds_local.py --no-warnings --export="write literal etree" --super="nyoka.PMML44Super" --subclass-suffix -o "nyoka.PMML44Super.py" -s "nyoka.PMML44.py" -b "behaviorsDir.xml" -f ..\nyoka.PMML44.xsd
#
# Current working directory (os.getcwd()):
#   PMML44
#

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

import sys
from lxml import etree as etree_

import nyoka.PMML44Super as supermod

def parsexml_(infile, parser=None, **kwargs):
    if parser is None:
        # Use the lxml ElementTree compatible parser so that, e.g.,
        #   we ignore comments.
        parser = etree_.ETCompatXMLParser(huge_tree=True)
    doc = etree_.parse(infile, parser=parser, **kwargs)
    return doc

#
# Globals
#

ExternalEncoding = 'utf-8'

#
# Data representation classes
#


class DefineFunction(supermod.DefineFunction):
    def __init__(self, name=None, optype=None, dataType=None, Extension=None, ParameterField=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None):
        super(DefineFunction, self).__init__(name, optype, dataType, Extension, ParameterField, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.DefineFunction.subclass = DefineFunction
# end class DefineFunction


class ParameterField(supermod.ParameterField):
    def __init__(self, name=None, optype=None, dataType=None, displayName=None):
        super(ParameterField, self).__init__(name, optype, dataType, displayName, )

    #
    # XMLBehaviors
    #
supermod.ParameterField.subclass = ParameterField
# end class ParameterField


class Apply(supermod.Apply):
    def __init__(self, function=None, mapMissingTo=None, defaultValue=None, invalidValueTreatment='returnInvalid', Extension=None, FieldRef=None, Apply_member=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None):
        super(Apply, self).__init__(function, mapMissingTo, defaultValue, invalidValueTreatment, Extension, FieldRef, Apply_member, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.Apply.subclass = Apply
# end class Apply


class MiningModel(supermod.MiningModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, Regression=None, DecisionTree=None, Segmentation=None, ModelVerification=None, Extension=None):
        super(MiningModel, self).__init__(modelName, functionName, algorithmName, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, Regression, DecisionTree, Segmentation, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.MiningModel.subclass = MiningModel
# end class MiningModel


class Segmentation(supermod.Segmentation):
    def __init__(self, multipleModelMethod=None, missingPredictionTreatment='continue', missingThreshold='1', Extension=None, Segment=None):
        super(Segmentation, self).__init__(multipleModelMethod, missingPredictionTreatment, missingThreshold, Extension, Segment, )

    #
    # XMLBehaviors
    #
supermod.Segmentation.subclass = Segmentation
# end class Segmentation


class Segment(supermod.Segment):
    def __init__(self, id=None, weight='1', Extension=None, SimplePredicate=None, CompoundPredicate=None, SimpleSetPredicate=None, True_=None, False_=None, AnomalyDetectionModel=None, AssociationModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, VariableWeight=None):
        super(Segment, self).__init__(id, weight, Extension, SimplePredicate, CompoundPredicate, SimpleSetPredicate, True_, False_, AnomalyDetectionModel, AssociationModel, BayesianNetworkModel, BaselineModel, ClusteringModel, GaussianProcessModel, GeneralRegressionModel, MiningModel, NaiveBayesModel, NearestNeighborModel, NeuralNetwork, RegressionModel, RuleSetModel, SequenceModel, Scorecard, SupportVectorMachineModel, TextModel, TimeSeriesModel, TreeModel, VariableWeight, )

    #
    # XMLBehaviors
    #
supermod.Segment.subclass = Segment
# end class Segment


class VariableWeight(supermod.VariableWeight):
    def __init__(self, field=None, Extension=None):
        super(VariableWeight, self).__init__(field, Extension, )

    #
    # XMLBehaviors
    #
supermod.VariableWeight.subclass = VariableWeight
# end class VariableWeight


class ResultField(supermod.ResultField):
    def __init__(self, name=None, displayName=None, optype=None, dataType=None, feature=None, value=None, Extension=None):
        super(ResultField, self).__init__(name, displayName, optype, dataType, feature, value, Extension, )

    #
    # XMLBehaviors
    #
supermod.ResultField.subclass = ResultField
# end class ResultField


class Regression(supermod.Regression):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, normalizationMethod='none', Extension=None, Output=None, ModelStats=None, Targets=None, LocalTransformations=None, ResultField=None, RegressionTable=None):
        super(Regression, self).__init__(modelName, functionName, algorithmName, normalizationMethod, Extension, Output, ModelStats, Targets, LocalTransformations, ResultField, RegressionTable, )

    #
    # XMLBehaviors
    #
supermod.Regression.subclass = Regression
# end class Regression


class DecisionTree(supermod.DecisionTree):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, missingValueStrategy='none', missingValuePenalty='1.0', noTrueChildStrategy='returnNullPrediction', splitCharacteristic='multiSplit', Extension=None, Output=None, ModelStats=None, Targets=None, LocalTransformations=None, ResultField=None, Node=None):
        super(DecisionTree, self).__init__(modelName, functionName, algorithmName, missingValueStrategy, missingValuePenalty, noTrueChildStrategy, splitCharacteristic, Extension, Output, ModelStats, Targets, LocalTransformations, ResultField, Node, )

    #
    # XMLBehaviors
    #
supermod.DecisionTree.subclass = DecisionTree
# end class DecisionTree


class SupportVectorMachineModel(supermod.SupportVectorMachineModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, threshold='0', svmRepresentation='SupportVectors', classificationMethod='OneAgainstAll', maxWins=False, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, LinearKernelType=None, PolynomialKernelType=None, RadialBasisKernelType=None, SigmoidKernelType=None, VectorDictionary=None, SupportVectorMachine=None, ModelVerification=None, Extension=None):
        super(SupportVectorMachineModel, self).__init__(modelName, functionName, algorithmName, threshold, svmRepresentation, classificationMethod, maxWins, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, LinearKernelType, PolynomialKernelType, RadialBasisKernelType, SigmoidKernelType, VectorDictionary, SupportVectorMachine, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.SupportVectorMachineModel.subclass = SupportVectorMachineModel
# end class SupportVectorMachineModel


class LinearKernelType(supermod.LinearKernelType):
    def __init__(self, description=None, Extension=None):
        super(LinearKernelType, self).__init__(description, Extension, )

    #
    # XMLBehaviors
    #
supermod.LinearKernelType.subclass = LinearKernelType
# end class LinearKernelType


class PolynomialKernelType(supermod.PolynomialKernelType):
    def __init__(self, description=None, gamma='1', coef0='1', degree='1', Extension=None):
        super(PolynomialKernelType, self).__init__(description, gamma, coef0, degree, Extension, )

    #
    # XMLBehaviors
    #
supermod.PolynomialKernelType.subclass = PolynomialKernelType
# end class PolynomialKernelType


class RadialBasisKernelType(supermod.RadialBasisKernelType):
    def __init__(self, description=None, gamma='1', Extension=None):
        super(RadialBasisKernelType, self).__init__(description, gamma, Extension, )

    #
    # XMLBehaviors
    #
supermod.RadialBasisKernelType.subclass = RadialBasisKernelType
# end class RadialBasisKernelType


class SigmoidKernelType(supermod.SigmoidKernelType):
    def __init__(self, description=None, gamma='1', coef0='1', Extension=None):
        super(SigmoidKernelType, self).__init__(description, gamma, coef0, Extension, )

    #
    # XMLBehaviors
    #
supermod.SigmoidKernelType.subclass = SigmoidKernelType
# end class SigmoidKernelType


class VectorDictionary(supermod.VectorDictionary):
    def __init__(self, numberOfVectors=None, Extension=None, VectorFields=None, VectorInstance=None):
        super(VectorDictionary, self).__init__(numberOfVectors, Extension, VectorFields, VectorInstance, )

    #
    # XMLBehaviors
    #
    def set_VectorInstance(self, VectorInstance, *args):
        self.VectorInstance = VectorInstance

        self.numberOfVectors = len(self.VectorInstance)
    def set_VectorInstance_wrapper(self, VectorInstance, *args):
        result = self.set_VectorInstance(VectorInstance, *args)
        return result

    def add_VectorInstance(self, value, *args):
        self.VectorInstance.append(value)

        self.numberOfVectors = len(self.VectorInstance)
    def add_VectorInstance_wrapper(self, value, *args):
        result = self.add_VectorInstance(value, *args)
        return result

    def insert_VectorInstance_at(self, index, value, *args):
        self.VectorInstance.insert(index, value)

        self.numberOfVectors = len(self.VectorInstance)
    def insert_VectorInstance_at_wrapper(self, index, value, *args):
        result = self.insert_VectorInstance_at(index, value, *args)
        return result

supermod.VectorDictionary.subclass = VectorDictionary
# end class VectorDictionary


class VectorFields(supermod.VectorFields):
    def __init__(self, numberOfFields=None, Extension=None, FieldRef=None, CategoricalPredictor=None):
        super(VectorFields, self).__init__(numberOfFields, Extension, FieldRef, CategoricalPredictor, )

    #
    # XMLBehaviors
    #
supermod.VectorFields.subclass = VectorFields
# end class VectorFields


class VectorInstance(supermod.VectorInstance):
    def __init__(self, id=None, Extension=None, REAL_SparseArray=None, Array=None):
        super(VectorInstance, self).__init__(id, Extension, REAL_SparseArray, Array, )

    #
    # XMLBehaviors
    #
supermod.VectorInstance.subclass = VectorInstance
# end class VectorInstance


class SupportVectorMachine(supermod.SupportVectorMachine):
    def __init__(self, targetCategory=None, alternateTargetCategory=None, threshold=None, Extension=None, SupportVectors=None, Coefficients=None):
        super(SupportVectorMachine, self).__init__(targetCategory, alternateTargetCategory, threshold, Extension, SupportVectors, Coefficients, )

    #
    # XMLBehaviors
    #
supermod.SupportVectorMachine.subclass = SupportVectorMachine
# end class SupportVectorMachine


class SupportVectors(supermod.SupportVectors):
    def __init__(self, numberOfSupportVectors=None, numberOfAttributes=None, Extension=None, SupportVector=None):
        super(SupportVectors, self).__init__(numberOfSupportVectors, numberOfAttributes, Extension, SupportVector, )

    #
    # XMLBehaviors
    #
    def set_SupportVector(self, SupportVector, *args):
        self.SupportVector = SupportVector

        self.numberOfVectors = len(self.SupportVector)
    def set_SupportVector_wrapper(self, SupportVector, *args):
        result = self.set_SupportVector(SupportVector, *args)
        return result

    def add_SupportVector(self, value, *args):
        self.SupportVector.append(value)

        self.numberOfVectors = len(self.SupportVector)
    def add_SupportVector_wrapper(self, value, *args):
        result = self.add_SupportVector(value, *args)
        return result

    def insert_SupportVector_at(self, index, value, *args):
        self.SupportVector.insert(index, value)

        self.numberOfVectors = len(self.SupportVector)
    def insert_SupportVector_at_wrapper(self, index, value, *args):
        result = self.insert_SupportVector_at(index, value, *args)
        return result

supermod.SupportVectors.subclass = SupportVectors
# end class SupportVectors


class SupportVector(supermod.SupportVector):
    def __init__(self, vectorId=None, Extension=None):
        super(SupportVector, self).__init__(vectorId, Extension, )

    #
    # XMLBehaviors
    #
supermod.SupportVector.subclass = SupportVector
# end class SupportVector


class Coefficients(supermod.Coefficients):
    def __init__(self, numberOfCoefficients=None, absoluteValue='0', Extension=None, Coefficient=None):
        super(Coefficients, self).__init__(numberOfCoefficients, absoluteValue, Extension, Coefficient, )

    #
    # XMLBehaviors
    #
    def set_Coefficient(self, Coefficient, *args):
        self.Coefficient = Coefficient

        self.numberOfCoefficients = len(self.Coefficient)
    def set_Coefficient_wrapper(self, Coefficient, *args):
        result = self.set_Coefficient(Coefficient, *args)
        return result

    def add_Coefficient(self, value, *args):
        self.Coefficient.append(value)

        self.numberOfCoefficients = len(self.Coefficient)
    def add_Coefficient_wrapper(self, value, *args):
        result = self.add_Coefficient(value, *args)
        return result

    def insert_Coefficient_at(self, index, value, *args):
        self.Coefficient.insert(index, value)

        self.numberOfCoefficients = len(self.Coefficient)
    def insert_Coefficient_at_wrapper(self, index, value, *args):
        result = self.insert_Coefficient_at(index, value, *args)
        return result

supermod.Coefficients.subclass = Coefficients
# end class Coefficients


class Coefficient(supermod.Coefficient):
    def __init__(self, value='0', Extension=None):
        super(Coefficient, self).__init__(value, Extension, )

    #
    # XMLBehaviors
    #
supermod.Coefficient.subclass = Coefficient
# end class Coefficient


class PMML(supermod.PMML):
    def __init__(self, version=None, Header=None, MiningBuildTask=None, DataDictionary=None, TransformationDictionary=None, AnomalyDetectionModel=None, AssociationModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, Extension=None):
        super(PMML, self).__init__(version, Header, MiningBuildTask, DataDictionary, TransformationDictionary, AnomalyDetectionModel, AssociationModel, BayesianNetworkModel, BaselineModel, ClusteringModel, GaussianProcessModel, GeneralRegressionModel, MiningModel, NaiveBayesModel, NearestNeighborModel, NeuralNetwork, RegressionModel, RuleSetModel, SequenceModel, Scorecard, SupportVectorMachineModel, TextModel, TimeSeriesModel, TreeModel, Extension, )

    #
    # XMLBehaviors
    #
    def export(self, outfile, level, namespace_='', name_='PMML', namespacedef_='', pretty_print=True, *args):
        imported_ns_def_ = supermod.GenerateDSNamespaceDefs_.get('Timestamp')

        if imported_ns_def_ is not None:

            namespacedef_ = imported_ns_def_

        if pretty_print:

            eol_ = '\n'

        else:

            eol_ = ''

        if self.original_tagname_ is not None:

            name_ = self.original_tagname_

        supermod.showIndent(outfile, level, pretty_print)

        outfile.write('<?xml version="1.0" encoding="UTF-8"?>' + eol_)

        outfile.write('<%s%s%s' % (namespace_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))

        already_processed = set()

        outfile.write(' xmlns="http://www.dmg.org/PMML-4_4"')

        self.exportAttributes(outfile, level, already_processed, namespace_, name_='Timestamp')

        if self.hasContent_():

            outfile.write('>%s' % (eol_, ))

            self.exportChildren(outfile, level + 1, namespace_='', name_='Timestamp', pretty_print=pretty_print)

            supermod.showIndent(outfile, 0, pretty_print)

            outfile.write('</%s%s>%s' % (namespace_, name_, eol_))

        else:

            outfile.write('/>%s' % (eol_, ))

        
    def export_wrapper(self, outfile, level, namespace_='', name_='PMML', namespacedef_='', pretty_print=True, *args):
        result = self.export(outfile, level, namespace_='', name_='PMML', namespacedef_='', pretty_print=True, *args)
        return result

supermod.PMML.subclass = PMML
# end class PMML


class MiningBuildTask(supermod.MiningBuildTask):
    def __init__(self, Extension=None):
        super(MiningBuildTask, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.MiningBuildTask.subclass = MiningBuildTask
# end class MiningBuildTask


class Extension(supermod.Extension):
    def __init__(self, extender=None, name=None, value=None, anytypeobjs_=None):
        super(Extension, self).__init__(extender, name, value, anytypeobjs_, )

    #
    # XMLBehaviors
    #
    def build(self, node, *args):
        already_processed = set()

        self.buildAttributes(node, node.attrib, already_processed)

        for child in node:

            nodeName_ = supermod.Tag_pattern_.match(child.tag).groups()[-1]

            self.buildChildren(child, node, nodeName_)

        if self.anytypeobjs_ == []:

            if node.text is not None:

                self.anytypeobjs_ = list(filter(None, [obj_.lstrip(' ') for obj_ in node.text.split('\n')]))

        return self
    def build_wrapper(self, node, *args):
        result = self.build(node, *args)
        return result

    def exportChildren(self, outfile, level, namespace_='', name_='Extension', fromsubclass_=False, pretty_print=True, *args):
        if pretty_print:

            eol_ = '\n'

        else:

            eol_ = ''

        for obj_ in self.anytypeobjs_:

            try:

                obj_.export(outfile, level, namespace_, pretty_print=pretty_print)

            except:

                showIndent(outfile, level, pretty_print)

                outfile.write(str(obj_))

                outfile.write(eol_)

        for objName_ in self.elementobjs_:

            obj_ = eval("self." + objName_)

            if eval("isinstance(obj_, list)"):

                for s in obj_:

                    showIndent(outfile, level, pretty_print)

                    outfile.write("<" + objName_ + ">" + str(s) + "</" + objName_ + ">")

                    outfile.write(eol_)

            else:

                showIndent(outfile, level, pretty_print)

                outfile.write("<" + objName_ + ">" + str(obj_) + "</" + objName_ + ">")

                outfile.write(eol_)
    def exportChildren_wrapper(self, outfile, level, namespace_='', name_='Extension', fromsubclass_=False, pretty_print=True, *args):
        result = self.exportChildren(outfile, level, namespace_='', name_='Extension', fromsubclass_=False, pretty_print=True, *args)
        return result

supermod.Extension.subclass = Extension
# end class Extension


class ArrayType(supermod.ArrayType):
    def __init__(self, n=None, type_=None, valueOf_=None, mixedclass_=None, content_=None):
        super(ArrayType, self).__init__(n, type_, valueOf_, mixedclass_, content_, )

    #
    # XMLBehaviors
    #
    def export(self, outfile, level, namespace_='', name_='ArrayType', namespacedef_='', pretty_print=True, *args):
        imported_ns_def_ = supermod.GenerateDSNamespaceDefs_.get('ArrayType')

        if imported_ns_def_ is not None:

            namespacedef_ = imported_ns_def_

        if pretty_print:

            eol_ = '\n'

        else:

            eol_ = ''

        if self.original_tagname_ is not None:

            name_ = self.original_tagname_

        supermod.showIndent(outfile, level, pretty_print)

        outfile.write('<%s%s%s' % (namespace_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))

        already_processed = set()

        self.exportAttributes(outfile, level, already_processed, namespace_, name_='ArrayType')

        if self.hasContent_():

            outfile.write('>')

            if not pretty_print:

                self.content_[0].value = self.content_[0].value.replace('\t', '').replace(' ', '')

                self.valueOf_ = self.valueOf_.replace('\t', '').replace(' ', '')

            self.exportChildren(outfile, level + 1, namespace_='', name_='ArrayType', pretty_print=pretty_print)

            # outfile.write(eol_)

            # supermod.showIndent(outfile, level, pretty_print)

            outfile.write('</%s%s>%s' % (namespace_, name_, eol_))

        else:

            outfile.write('/>%s' % (eol_, ))
    def export_wrapper(self, outfile, level, namespace_='', name_='ArrayType', namespacedef_='', pretty_print=True, *args):
        result = self.export(outfile, level, namespace_='', name_='ArrayType', namespacedef_='', pretty_print=True, *args)
        return result

    def exportChildren(self, outfile, level, namespace_='', name_='ArrayType', fromsubclass_=False, pretty_print=True, *args):
        if not fromsubclass_:

            for item_ in self.content_:

                item_.export(outfile, level, item_.name, namespace_, pretty_print=pretty_print)

                
    def exportChildren_wrapper(self, outfile, level, namespace_='', name_='ArrayType', fromsubclass_=False, pretty_print=True, *args):
        result = self.exportChildren(outfile, level, namespace_='', name_='ArrayType', fromsubclass_=False, pretty_print=True, *args)
        return result

supermod.ArrayType.subclass = ArrayType
# end class ArrayType


class INT_SparseArray(supermod.INT_SparseArray):
    def __init__(self, n=None, defaultValue='0', Indices=None, INT_Entries=None):
        super(INT_SparseArray, self).__init__(n, defaultValue, Indices, INT_Entries, )

    #
    # XMLBehaviors
    #
supermod.INT_SparseArray.subclass = INT_SparseArray
# end class INT_SparseArray


class REAL_SparseArray(supermod.REAL_SparseArray):
    def __init__(self, n=None, defaultValue='0', Indices=None, REAL_Entries=None):
        super(REAL_SparseArray, self).__init__(n, defaultValue, Indices, REAL_Entries, )

    #
    # XMLBehaviors
    #
supermod.REAL_SparseArray.subclass = REAL_SparseArray
# end class REAL_SparseArray


class Matrix(supermod.Matrix):
    def __init__(self, kind='any', nbRows=None, nbCols=None, diagDefault=None, offDiagDefault=None, Array=None, MatCell=None):
        super(Matrix, self).__init__(kind, nbRows, nbCols, diagDefault, offDiagDefault, Array, MatCell, )

    #
    # XMLBehaviors
    #
supermod.Matrix.subclass = Matrix
# end class Matrix


class MatCell(supermod.MatCell):
    def __init__(self, row=None, col=None, valueOf_=None):
        super(MatCell, self).__init__(row, col, valueOf_, )

    #
    # XMLBehaviors
    #
supermod.MatCell.subclass = MatCell
# end class MatCell


class RegressionModel(supermod.RegressionModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, modelType=None, targetFieldName=None, normalizationMethod='none', isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, RegressionTable=None, ModelVerification=None, Extension=None):
        super(RegressionModel, self).__init__(modelName, functionName, algorithmName, modelType, targetFieldName, normalizationMethod, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, RegressionTable, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.RegressionModel.subclass = RegressionModel
# end class RegressionModel


class RegressionTable(supermod.RegressionTable):
    def __init__(self, intercept=None, targetCategory=None, Extension=None, NumericPredictor=None, CategoricalPredictor=None, PredictorTerm=None):
        super(RegressionTable, self).__init__(intercept, targetCategory, Extension, NumericPredictor, CategoricalPredictor, PredictorTerm, )

    #
    # XMLBehaviors
    #
supermod.RegressionTable.subclass = RegressionTable
# end class RegressionTable


class NumericPredictor(supermod.NumericPredictor):
    def __init__(self, name=None, exponent='1', coefficient=None, Extension=None):
        super(NumericPredictor, self).__init__(name, exponent, coefficient, Extension, )

    #
    # XMLBehaviors
    #
supermod.NumericPredictor.subclass = NumericPredictor
# end class NumericPredictor


class CategoricalPredictor(supermod.CategoricalPredictor):
    def __init__(self, name=None, value=None, coefficient=None, Extension=None):
        super(CategoricalPredictor, self).__init__(name, value, coefficient, Extension, )

    #
    # XMLBehaviors
    #
supermod.CategoricalPredictor.subclass = CategoricalPredictor
# end class CategoricalPredictor


class PredictorTerm(supermod.PredictorTerm):
    def __init__(self, name=None, coefficient=None, Extension=None, FieldRef=None):
        super(PredictorTerm, self).__init__(name, coefficient, Extension, FieldRef, )

    #
    # XMLBehaviors
    #
supermod.PredictorTerm.subclass = PredictorTerm
# end class PredictorTerm


class RuleSetModel(supermod.RuleSetModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, RuleSet=None, ModelVerification=None, Extension=None):
        super(RuleSetModel, self).__init__(modelName, functionName, algorithmName, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, RuleSet, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.RuleSetModel.subclass = RuleSetModel
# end class RuleSetModel


class RuleSet(supermod.RuleSet):
    def __init__(self, recordCount=None, nbCorrect=None, defaultScore=None, defaultConfidence=None, Extension=None, RuleSelectionMethod=None, ScoreDistribution=None, SimpleRule=None, CompoundRule=None):
        super(RuleSet, self).__init__(recordCount, nbCorrect, defaultScore, defaultConfidence, Extension, RuleSelectionMethod, ScoreDistribution, SimpleRule, CompoundRule, )

    #
    # XMLBehaviors
    #
supermod.RuleSet.subclass = RuleSet
# end class RuleSet


class RuleSelectionMethod(supermod.RuleSelectionMethod):
    def __init__(self, criterion=None, Extension=None):
        super(RuleSelectionMethod, self).__init__(criterion, Extension, )

    #
    # XMLBehaviors
    #
supermod.RuleSelectionMethod.subclass = RuleSelectionMethod
# end class RuleSelectionMethod


class SimpleRule(supermod.SimpleRule):
    def __init__(self, id=None, score=None, recordCount=None, nbCorrect=None, confidence='1', weight='1', Extension=None, SimplePredicate=None, CompoundPredicate=None, SimpleSetPredicate=None, True_=None, False_=None, ScoreDistribution=None):
        super(SimpleRule, self).__init__(id, score, recordCount, nbCorrect, confidence, weight, Extension, SimplePredicate, CompoundPredicate, SimpleSetPredicate, True_, False_, ScoreDistribution, )

    #
    # XMLBehaviors
    #
supermod.SimpleRule.subclass = SimpleRule
# end class SimpleRule


class CompoundRule(supermod.CompoundRule):
    def __init__(self, Extension=None, SimplePredicate=None, CompoundPredicate=None, SimpleSetPredicate=None, True_=None, False_=None, SimpleRule=None, CompoundRule_member=None):
        super(CompoundRule, self).__init__(Extension, SimplePredicate, CompoundPredicate, SimpleSetPredicate, True_, False_, SimpleRule, CompoundRule_member, )

    #
    # XMLBehaviors
    #
supermod.CompoundRule.subclass = CompoundRule
# end class CompoundRule


class ModelExplanation(supermod.ModelExplanation):
    def __init__(self, Extension=None, PredictiveModelQuality=None, ClusteringModelQuality=None, Correlations=None):
        super(ModelExplanation, self).__init__(Extension, PredictiveModelQuality, ClusteringModelQuality, Correlations, )

    #
    # XMLBehaviors
    #
supermod.ModelExplanation.subclass = ModelExplanation
# end class ModelExplanation


class PredictiveModelQuality(supermod.PredictiveModelQuality):
    def __init__(self, targetField=None, dataName=None, dataUsage='training', meanError=None, meanAbsoluteError=None, meanSquaredError=None, rootMeanSquaredError=None, r_squared=None, adj_r_squared=None, sumSquaredError=None, sumSquaredRegression=None, numOfRecords=None, numOfRecordsWeighted=None, numOfPredictors=None, degreesOfFreedom=None, fStatistic=None, AIC=None, BIC=None, AICc=None, accuracy=None, AUC=None, precision=None, recall=None, specificity=None, F1=None, F2=None, Fhalf=None, Extension=None, ConfusionMatrix=None, LiftData=None, ROC=None):
        super(PredictiveModelQuality, self).__init__(targetField, dataName, dataUsage, meanError, meanAbsoluteError, meanSquaredError, rootMeanSquaredError, r_squared, adj_r_squared, sumSquaredError, sumSquaredRegression, numOfRecords, numOfRecordsWeighted, numOfPredictors, degreesOfFreedom, fStatistic, AIC, BIC, AICc, accuracy, AUC, precision, recall, specificity, F1, F2, Fhalf, Extension, ConfusionMatrix, LiftData, ROC, )

    #
    # XMLBehaviors
    #
supermod.PredictiveModelQuality.subclass = PredictiveModelQuality
# end class PredictiveModelQuality


class ClusteringModelQuality(supermod.ClusteringModelQuality):
    def __init__(self, dataName=None, SSE=None, SSB=None, Extension=None):
        super(ClusteringModelQuality, self).__init__(dataName, SSE, SSB, Extension, )

    #
    # XMLBehaviors
    #
supermod.ClusteringModelQuality.subclass = ClusteringModelQuality
# end class ClusteringModelQuality


class LiftData(supermod.LiftData):
    def __init__(self, targetFieldValue=None, targetFieldDisplayValue=None, rankingQuality=None, Extension=None, ModelLiftGraph=None, OptimumLiftGraph=None, RandomLiftGraph=None):
        super(LiftData, self).__init__(targetFieldValue, targetFieldDisplayValue, rankingQuality, Extension, ModelLiftGraph, OptimumLiftGraph, RandomLiftGraph, )

    #
    # XMLBehaviors
    #
supermod.LiftData.subclass = LiftData
# end class LiftData


class ModelLiftGraph(supermod.ModelLiftGraph):
    def __init__(self, Extension=None, LiftGraph=None):
        super(ModelLiftGraph, self).__init__(Extension, LiftGraph, )

    #
    # XMLBehaviors
    #
supermod.ModelLiftGraph.subclass = ModelLiftGraph
# end class ModelLiftGraph


class OptimumLiftGraph(supermod.OptimumLiftGraph):
    def __init__(self, Extension=None, LiftGraph=None):
        super(OptimumLiftGraph, self).__init__(Extension, LiftGraph, )

    #
    # XMLBehaviors
    #
supermod.OptimumLiftGraph.subclass = OptimumLiftGraph
# end class OptimumLiftGraph


class RandomLiftGraph(supermod.RandomLiftGraph):
    def __init__(self, Extension=None, LiftGraph=None):
        super(RandomLiftGraph, self).__init__(Extension, LiftGraph, )

    #
    # XMLBehaviors
    #
supermod.RandomLiftGraph.subclass = RandomLiftGraph
# end class RandomLiftGraph


class LiftGraph(supermod.LiftGraph):
    def __init__(self, Extension=None, XCoordinates=None, YCoordinates=None, BoundaryValues=None, BoundaryValueMeans=None):
        super(LiftGraph, self).__init__(Extension, XCoordinates, YCoordinates, BoundaryValues, BoundaryValueMeans, )

    #
    # XMLBehaviors
    #
supermod.LiftGraph.subclass = LiftGraph
# end class LiftGraph


class XCoordinates(supermod.XCoordinates):
    def __init__(self, Extension=None, Array=None):
        super(XCoordinates, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.XCoordinates.subclass = XCoordinates
# end class XCoordinates


class YCoordinates(supermod.YCoordinates):
    def __init__(self, Extension=None, Array=None):
        super(YCoordinates, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.YCoordinates.subclass = YCoordinates
# end class YCoordinates


class BoundaryValues(supermod.BoundaryValues):
    def __init__(self, Extension=None, Array=None):
        super(BoundaryValues, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.BoundaryValues.subclass = BoundaryValues
# end class BoundaryValues


class BoundaryValueMeans(supermod.BoundaryValueMeans):
    def __init__(self, Extension=None, Array=None):
        super(BoundaryValueMeans, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.BoundaryValueMeans.subclass = BoundaryValueMeans
# end class BoundaryValueMeans


class ROC(supermod.ROC):
    def __init__(self, positiveTargetFieldValue=None, positiveTargetFieldDisplayValue=None, negativeTargetFieldValue=None, negativeTargetFieldDisplayValue=None, Extension=None, ROCGraph=None):
        super(ROC, self).__init__(positiveTargetFieldValue, positiveTargetFieldDisplayValue, negativeTargetFieldValue, negativeTargetFieldDisplayValue, Extension, ROCGraph, )

    #
    # XMLBehaviors
    #
supermod.ROC.subclass = ROC
# end class ROC


class ROCGraph(supermod.ROCGraph):
    def __init__(self, Extension=None, XCoordinates=None, YCoordinates=None, BoundaryValues=None):
        super(ROCGraph, self).__init__(Extension, XCoordinates, YCoordinates, BoundaryValues, )

    #
    # XMLBehaviors
    #
supermod.ROCGraph.subclass = ROCGraph
# end class ROCGraph


class ConfusionMatrix(supermod.ConfusionMatrix):
    def __init__(self, Extension=None, ClassLabels=None, Matrix=None):
        super(ConfusionMatrix, self).__init__(Extension, ClassLabels, Matrix, )

    #
    # XMLBehaviors
    #
supermod.ConfusionMatrix.subclass = ConfusionMatrix
# end class ConfusionMatrix


class ClassLabels(supermod.ClassLabels):
    def __init__(self, Extension=None, Array=None):
        super(ClassLabels, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.ClassLabels.subclass = ClassLabels
# end class ClassLabels


class Correlations(supermod.Correlations):
    def __init__(self, Extension=None, CorrelationFields=None, CorrelationValues=None, CorrelationMethods=None):
        super(Correlations, self).__init__(Extension, CorrelationFields, CorrelationValues, CorrelationMethods, )

    #
    # XMLBehaviors
    #
supermod.Correlations.subclass = Correlations
# end class Correlations


class CorrelationFields(supermod.CorrelationFields):
    def __init__(self, Extension=None, Array=None):
        super(CorrelationFields, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.CorrelationFields.subclass = CorrelationFields
# end class CorrelationFields


class CorrelationValues(supermod.CorrelationValues):
    def __init__(self, Extension=None, Matrix=None):
        super(CorrelationValues, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.CorrelationValues.subclass = CorrelationValues
# end class CorrelationValues


class CorrelationMethods(supermod.CorrelationMethods):
    def __init__(self, Extension=None, Matrix=None):
        super(CorrelationMethods, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.CorrelationMethods.subclass = CorrelationMethods
# end class CorrelationMethods


class Taxonomy(supermod.Taxonomy):
    def __init__(self, name=None, Extension=None, ChildParent=None):
        super(Taxonomy, self).__init__(name, Extension, ChildParent, )

    #
    # XMLBehaviors
    #
supermod.Taxonomy.subclass = Taxonomy
# end class Taxonomy


class ChildParent(supermod.ChildParent):
    def __init__(self, childField=None, parentField=None, parentLevelField=None, isRecursive='no', Extension=None, FieldColumnPair=None, TableLocator=None, InlineTable=None):
        super(ChildParent, self).__init__(childField, parentField, parentLevelField, isRecursive, Extension, FieldColumnPair, TableLocator, InlineTable, )

    #
    # XMLBehaviors
    #
supermod.ChildParent.subclass = ChildParent
# end class ChildParent


class TableLocator(supermod.TableLocator):
    def __init__(self, Extension=None):
        super(TableLocator, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.TableLocator.subclass = TableLocator
# end class TableLocator


class InlineTable(supermod.InlineTable):
    def __init__(self, Extension=None, row=None):
        super(InlineTable, self).__init__(Extension, row, )

    #
    # XMLBehaviors
    #
supermod.InlineTable.subclass = InlineTable
# end class InlineTable


class row(supermod.row):
    def __init__(self, anytypeobjs_=None):
        super(row, self).__init__(anytypeobjs_, )

    #
    # XMLBehaviors
    #
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, *args):
        if not hasattr(self, "elementobjs_"):

            self.elementobjs_ = []

        if hasattr(self, nodeName_) and nodeName_ not in self.elementobjs_:

            nodeName_ += '_'

        if nodeName_ not in self.elementobjs_:

            self.elementobjs_.append(nodeName_)

        if not eval("hasattr(self, '" + nodeName_ + "')"):

            nodeVal = list(filter(None, [obj_.lstrip(' ') for obj_ in child_.text.split('\n')]))[0]

            try:

                setattr(self, nodeName_,eval(nodeVal))

            except:

                setattr(self, nodeName_,nodeVal)

        else:

            if getattr(self,nodeName_).__class__.__name__ == 'str':

                setattr(self,nodeName_,[getattr(self,nodeName_)])

            else:

                setattr(self,nodeName_,list(getattr(self,nodeName_)))

            nodeVal = list(filter(None, [obj_.lstrip(' ') for obj_ in child_.text.split('\n')]))[0]

            try:

                getattr(self, nodeName_).append(eval(nodeVal))

            except:

                getattr(self, nodeName_).append(nodeVal)

                
    def buildChildren_wrapper(self, child_, node, nodeName_, fromsubclass_=False, *args):
        result = self.buildChildren(child_, node, nodeName_, fromsubclass_=False, *args)
        return result

supermod.row.subclass = row
# end class row


class AssociationModel(supermod.AssociationModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, numberOfTransactions=None, maxNumberOfItemsPerTA=None, avgNumberOfItemsPerTA=None, minimumSupport=None, minimumConfidence=None, lengthLimit=None, numberOfItems=None, numberOfItemsets=None, numberOfRules=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, LocalTransformations=None, Item=None, Itemset=None, AssociationRule=None, ModelVerification=None, Extension=None):
        super(AssociationModel, self).__init__(modelName, functionName, algorithmName, numberOfTransactions, maxNumberOfItemsPerTA, avgNumberOfItemsPerTA, minimumSupport, minimumConfidence, lengthLimit, numberOfItems, numberOfItemsets, numberOfRules, isScorable, MiningSchema, Output, ModelStats, LocalTransformations, Item, Itemset, AssociationRule, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
    def set_Item(self, Item, *args):
        self.Item = Item

        self.numberOfItems = len(self.Item)
    def set_Item_wrapper(self, Item, *args):
        result = self.set_Item(Item, *args)
        return result

    def add_Item(self, value, *args):
        self.Item.append(value)

        self.numberOfItems = len(self.Item)
    def add_Item_wrapper(self, value, *args):
        result = self.add_Item(value, *args)
        return result

    def insert_Item_at(self, index, value, *args):
        self.Item.insert(index, value)

        self.numberOfItems = len(self.Item)
    def insert_Item_at_wrapper(self, index, value, *args):
        result = self.insert_Item_at(index, value, *args)
        return result

    def set_Itemset(self, Itemset, *args):
        self.Itemset = Itemset

        self.numberOfItemsets = len(self.Itemset)
    def set_Itemset_wrapper(self, Itemset, *args):
        result = self.set_Itemset(Itemset, *args)
        return result

    def add_Itemset(self, value, *args):
        self.Itemset.append(value)

        self.numberOfItemsets = len(self.Itemset)
    def add_Itemset_wrapper(self, value, *args):
        result = self.add_Itemset(value, *args)
        return result

    def insert_Itemset_at(self, index, value, *args):
        self.Itemset.insert(index, value)

        self.numberOfItemsets = len(self.Itemset)
    def insert_Itemset_at_wrapper(self, index, value, *args):
        result = self.insert_Itemset_at(index, value, *args)
        return result

    def set_AssociationRule(self, Rules, *args):
        pass

    def set_AssociationRule_wrapper(self, Rules, *args):
        result = self.set_AssociationRule(Rules, *args)
        return result

    def add_AssociationRule(self, value, *args):
        self.AssociationRule.append(value)

        self.numberOfRules = len(self.AssociationRule)
    def add_AssociationRule_wrapper(self, value, *args):
        result = self.add_AssociationRule(value, *args)
        return result

    def insert_AssociationRule_at(self, index, value, *args):
        self.AssociationRule.insert(index, value)

        self.numberOfRules = len(self.AssociationRule)
    def insert_AssociationRule_at_wrapper(self, index, value, *args):
        result = self.insert_AssociationRule_at(index, value, *args)
        return result

supermod.AssociationModel.subclass = AssociationModel
# end class AssociationModel


class Item(supermod.Item):
    def __init__(self, id=None, value=None, field=None, category=None, mappedValue=None, weight=None, Extension=None):
        super(Item, self).__init__(id, value, field, category, mappedValue, weight, Extension, )

    #
    # XMLBehaviors
    #
supermod.Item.subclass = Item
# end class Item


class Itemset(supermod.Itemset):
    def __init__(self, id=None, support=None, numberOfItems=None, Extension=None, ItemRef=None):
        super(Itemset, self).__init__(id, support, numberOfItems, Extension, ItemRef, )

    #
    # XMLBehaviors
    #
    def set_ItemRef(self, ItemRef, *args):
        self.ItemRef = ItemRef

        self.numberOfItems = len(self.ItemRef)
    def set_ItemRef_wrapper(self, ItemRef, *args):
        result = self.set_ItemRef(ItemRef, *args)
        return result

    def add_ItemRef(self, value, *args):
        self.ItemRef.append(value)

        self.numberOfItems = len(self.ItemRef)
    def add_ItemRef_wrapper(self, value, *args):
        result = self.add_ItemRef(value, *args)
        return result

    def insert_ItemRef_at(self, index, value, *args):
        self.ItemRef.insert(index, value)

        self.numberOfItems = len(self.ItemRef)
    def insert_ItemRef_at_wrapper(self, index, value, *args):
        result = self.insert_ItemRef_at(index, value, *args)
        return result

supermod.Itemset.subclass = Itemset
# end class Itemset


class ItemRef(supermod.ItemRef):
    def __init__(self, itemRef=None, Extension=None):
        super(ItemRef, self).__init__(itemRef, Extension, )

    #
    # XMLBehaviors
    #
supermod.ItemRef.subclass = ItemRef
# end class ItemRef


class AssociationRule(supermod.AssociationRule):
    def __init__(self, antecedent=None, consequent=None, support=None, confidence=None, lift=None, leverage=None, affinity=None, id=None, Extension=None):
        super(AssociationRule, self).__init__(antecedent, consequent, support, confidence, lift, leverage, affinity, id, Extension, )

    #
    # XMLBehaviors
    #
supermod.AssociationRule.subclass = AssociationRule
# end class AssociationRule


class MiningSchema(supermod.MiningSchema):
    def __init__(self, Extension=None, MiningField=None):
        super(MiningSchema, self).__init__(Extension, MiningField, )

    #
    # XMLBehaviors
    #
supermod.MiningSchema.subclass = MiningSchema
# end class MiningSchema


class MiningField(supermod.MiningField):
    def __init__(self, name=None, usageType='active', optype=None, importance=None, outliers='asIs', lowValue=None, highValue=None, missingValueReplacement=None, missingValueTreatment=None, invalidValueTreatment='returnInvalid', invalidValueReplacement=None, Extension=None):
        super(MiningField, self).__init__(name, usageType, optype, importance, outliers, lowValue, highValue, missingValueReplacement, missingValueTreatment, invalidValueTreatment, invalidValueReplacement, Extension, )

    #
    # XMLBehaviors
    #
    def exportAttributes(self, outfile, level, already_processed, namespace_='', name_='MiningField', *args):
        if self.name is not None and 'name' not in already_processed:

            already_processed.add('name')

            outfile.write(' name=%s' % (supermod.quote_attrib(self.name), ))

        if self.usageType is not None and 'usageType' not in already_processed:

            already_processed.add('usageType')

            outfile.write(' usageType=%s' % (supermod.quote_attrib(self.usageType), ))

        if self.optype is not None and 'optype' not in already_processed:

            already_processed.add('optype')

            outfile.write(' optype=%s' % (supermod.quote_attrib(self.optype), ))

        if self.importance is not None and 'importance' not in already_processed:

            already_processed.add('importance')

            outfile.write(' importance=%s' % (supermod.quote_attrib(self.importance), ))

        if self.outliers != "asIs" and 'outliers' not in already_processed:

            already_processed.add('outliers')

            outfile.write(' outliers=%s' % (supermod.quote_attrib(self.outliers), ))

        if self.lowValue is not None and 'lowValue' not in already_processed:

            already_processed.add('lowValue')

            outfile.write(' lowValue=%s' % (supermod.quote_attrib(self.lowValue), ))

        if self.highValue is not None and 'highValue' not in already_processed:

            already_processed.add('highValue')

            outfile.write(' highValue=%s' % (supermod.quote_attrib(self.highValue), ))

        if self.missingValueReplacement is not None and 'missingValueReplacement' not in already_processed:

            already_processed.add('missingValueReplacement')

            outfile.write(' missingValueReplacement=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.missingValueReplacement), input_name='missingValueReplacement')), ))

        if self.missingValueTreatment is not None and 'missingValueTreatment' not in already_processed:

            already_processed.add('missingValueTreatment')

            outfile.write(' missingValueTreatment=%s' % (supermod.quote_attrib(self.missingValueTreatment), ))

        if self.invalidValueTreatment != "returnInvalid" and 'invalidValueTreatment' not in already_processed:

            already_processed.add('invalidValueTreatment')

            outfile.write(' invalidValueTreatment=%s' % (supermod.quote_attrib(self.invalidValueTreatment), ))
    def exportAttributes_wrapper(self, outfile, level, already_processed, namespace_='', name_='MiningField', *args):
        result = self.exportAttributes(outfile, level, already_processed, namespace_='', name_='MiningField', *args)
        return result

supermod.MiningField.subclass = MiningField
# end class MiningField


class Output(supermod.Output):
    def __init__(self, Extension=None, OutputField=None):
        super(Output, self).__init__(Extension, OutputField, )

    #
    # XMLBehaviors
    #
supermod.Output.subclass = Output
# end class Output


class OutputField(supermod.OutputField):
    def __init__(self, name=None, displayName=None, optype=None, dataType=None, targetField=None, feature='predictedValue', value=None, ruleFeature='consequent', algorithm='exclusiveRecommendation', rank='1', rankBasis='confidence', rankOrder='descending', isMultiValued='0', segmentId=None, isFinalResult=True, Extension=None, Decisions=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None, Value=None):
        super(OutputField, self).__init__(name, displayName, optype, dataType, targetField, feature, value, ruleFeature, algorithm, rank, rankBasis, rankOrder, isMultiValued, segmentId, isFinalResult, Extension, Decisions, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, Value, )

    #
    # XMLBehaviors
    #
    def exportAttributes(self, outfile, level, already_processed, namespace_='', name_='OutputFields', *args):
        if self.name is not None and 'name' not in already_processed:

            already_processed.add('name')

            outfile.write(' name=%s' % (supermod.quote_attrib(self.name), ))

        if self.displayName is not None and 'displayName' not in already_processed:

            already_processed.add('displayName')

            outfile.write(' displayName=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.displayName), input_name='displayName')), ))

        if self.optype is not None and 'optype' not in already_processed:

            already_processed.add('optype')

            outfile.write(' optype=%s' % (supermod.quote_attrib(self.optype), ))

        if self.dataType is not None and 'dataType' not in already_processed:

            already_processed.add('dataType')

            outfile.write(' dataType=%s' % (supermod.quote_attrib(self.dataType), ))

        if self.targetField is not None and 'targetField' not in already_processed:

            already_processed.add('targetField')

            outfile.write(' targetField=%s' % (supermod.quote_attrib(self.targetField), ))

        if self.feature is not None and 'feature' not in already_processed:

            already_processed.add('feature')

            outfile.write(' feature=%s' % (supermod.quote_attrib(self.feature), ))

        if self.value is not None and 'value' not in already_processed:

            already_processed.add('value')

            outfile.write(' value=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.value), input_name='value')), ))

        if self.ruleFeature != "consequent" and 'ruleFeature' not in already_processed:

            already_processed.add('ruleFeature')

            outfile.write(' ruleFeature=%s' % (supermod.quote_attrib(self.ruleFeature), ))

        if self.algorithm != "exclusiveRecommendation" and 'algorithm' not in already_processed:

            already_processed.add('algorithm')

            outfile.write(' algorithm=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.algorithm), input_name='algorithm')), ))

        # if self.rank is not None and 'rank' not in already_processed:

            # already_processed.add('rank')

            # outfile.write(' rank=%s' % (supermod.quote_attrib(self.rank), ))

        if self.rankBasis != "confidence" and 'rankBasis' not in already_processed:

            already_processed.add('rankBasis')

            outfile.write(' rankBasis=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.rankBasis), input_name='rankBasis')), ))

        if self.rankOrder != "descending" and 'rankOrder' not in already_processed:

            already_processed.add('rankOrder')

            outfile.write(' rankOrder=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.rankOrder), input_name='rankOrder')), ))

        if self.isMultiValued != "0" and 'isMultiValued' not in already_processed:

            already_processed.add('isMultiValued')

            outfile.write(' isMultiValued=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.isMultiValued), input_name='isMultiValued')), ))

        if self.segmentId is not None and 'segmentId' not in already_processed:

            already_processed.add('segmentId')

            outfile.write(' segmentId=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.segmentId), input_name='segmentId')), ))

        if not self.isFinalResult and 'isFinalResult' not in already_processed:

            already_processed.add('isFinalResult')

            outfile.write(' isFinalResult="%s"' % self.gds_format_boolean(self.isFinalResult, input_name='isFinalResult'))


    def exportAttributes_wrapper(self, outfile, level, already_processed, namespace_='', name_='OutputFields', *args):
        result = self.exportAttributes(outfile, level, already_processed, namespace_='', name_='OutputFields', *args)
        return result

supermod.OutputField.subclass = OutputField
# end class OutputField


class Decisions(supermod.Decisions):
    def __init__(self, businessProblem=None, description=None, Extension=None, Decision=None):
        super(Decisions, self).__init__(businessProblem, description, Extension, Decision, )

    #
    # XMLBehaviors
    #
supermod.Decisions.subclass = Decisions
# end class Decisions


class Decision(supermod.Decision):
    def __init__(self, value=None, displayValue=None, description=None, Extension=None):
        super(Decision, self).__init__(value, displayValue, description, Extension, )

    #
    # XMLBehaviors
    #
supermod.Decision.subclass = Decision
# end class Decision


class TreeModel(supermod.TreeModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, missingValueStrategy='none', missingValuePenalty='1.0', noTrueChildStrategy='returnNullPrediction', splitCharacteristic='multiSplit', isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, Node=None, ModelVerification=None, Extension=None):
        super(TreeModel, self).__init__(modelName, functionName, algorithmName, missingValueStrategy, missingValuePenalty, noTrueChildStrategy, splitCharacteristic, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, Node, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.TreeModel.subclass = TreeModel
# end class TreeModel


class Node(supermod.Node):
    def __init__(self, id=None, score=None, recordCount=None, defaultChild=None, SimplePredicate=None, CompoundPredicate=None, SimpleSetPredicate=None, True_=None, False_=None, Partition=None, ScoreDistribution=None, Node_member=None, Extension=None, Regression=None, DecisionTree=None):
        super(Node, self).__init__(id, score, recordCount, defaultChild, SimplePredicate, CompoundPredicate, SimpleSetPredicate, True_, False_, Partition, ScoreDistribution, Node_member, Extension, Regression, DecisionTree, )

    #
    # XMLBehaviors
    #
supermod.Node.subclass = Node
# end class Node


class SimplePredicate(supermod.SimplePredicate):
    def __init__(self, field=None, operator=None, value=None, Extension=None):
        super(SimplePredicate, self).__init__(field, operator, value, Extension, )

    #
    # XMLBehaviors
    #
supermod.SimplePredicate.subclass = SimplePredicate
# end class SimplePredicate


class CompoundPredicate(supermod.CompoundPredicate):
    def __init__(self, booleanOperator=None, Extension=None, SimplePredicate=None, CompoundPredicate_member=None, SimpleSetPredicate=None, True_=None, False_=None):
        super(CompoundPredicate, self).__init__(booleanOperator, Extension, SimplePredicate, CompoundPredicate_member, SimpleSetPredicate, True_, False_, )

    #
    # XMLBehaviors
    #
supermod.CompoundPredicate.subclass = CompoundPredicate
# end class CompoundPredicate


class SimpleSetPredicate(supermod.SimpleSetPredicate):
    def __init__(self, field=None, booleanOperator=None, Extension=None, Array=None):
        super(SimpleSetPredicate, self).__init__(field, booleanOperator, Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.SimpleSetPredicate.subclass = SimpleSetPredicate
# end class SimpleSetPredicate


class True_(supermod.True_):
    def __init__(self, Extension=None):
        super(True_, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.True_.subclass = True_
# end class True_


class False_(supermod.False_):
    def __init__(self, Extension=None):
        super(False_, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.False_.subclass = False_
# end class False_


class ScoreDistribution(supermod.ScoreDistribution):
    def __init__(self, value=None, recordCount=None, confidence=None, probability=None, Extension=None):
        super(ScoreDistribution, self).__init__(value, recordCount, confidence, probability, Extension, )

    #
    # XMLBehaviors
    #
supermod.ScoreDistribution.subclass = ScoreDistribution
# end class ScoreDistribution


class Scorecard(supermod.Scorecard):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, initialScore='0', useReasonCodes=True, reasonCodeAlgorithm='pointsBelow', baselineScore=None, baselineMethod='other', isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, Characteristics=None, ModelVerification=None, Extension=None):
        super(Scorecard, self).__init__(modelName, functionName, algorithmName, initialScore, useReasonCodes, reasonCodeAlgorithm, baselineScore, baselineMethod, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, Characteristics, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.Scorecard.subclass = Scorecard
# end class Scorecard


class Characteristics(supermod.Characteristics):
    def __init__(self, Extension=None, Characteristic=None):
        super(Characteristics, self).__init__(Extension, Characteristic, )

    #
    # XMLBehaviors
    #
supermod.Characteristics.subclass = Characteristics
# end class Characteristics


class Characteristic(supermod.Characteristic):
    def __init__(self, name=None, reasonCode=None, baselineScore=None, Extension=None, Attribute=None):
        super(Characteristic, self).__init__(name, reasonCode, baselineScore, Extension, Attribute, )

    #
    # XMLBehaviors
    #
supermod.Characteristic.subclass = Characteristic
# end class Characteristic


class Attribute(supermod.Attribute):
    def __init__(self, reasonCode=None, partialScore=None, Extension=None, SimplePredicate=None, CompoundPredicate=None, SimpleSetPredicate=None, True_=None, False_=None, ComplexPartialScore=None):
        super(Attribute, self).__init__(reasonCode, partialScore, Extension, SimplePredicate, CompoundPredicate, SimpleSetPredicate, True_, False_, ComplexPartialScore, )

    #
    # XMLBehaviors
    #
supermod.Attribute.subclass = Attribute
# end class Attribute


class ComplexPartialScore(supermod.ComplexPartialScore):
    def __init__(self, Extension=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None):
        super(ComplexPartialScore, self).__init__(Extension, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.ComplexPartialScore.subclass = ComplexPartialScore
# end class ComplexPartialScore


class ModelStats(supermod.ModelStats):
    def __init__(self, Extension=None, UnivariateStats=None, MultivariateStats=None):
        super(ModelStats, self).__init__(Extension, UnivariateStats, MultivariateStats, )

    #
    # XMLBehaviors
    #
supermod.ModelStats.subclass = ModelStats
# end class ModelStats


class UnivariateStats(supermod.UnivariateStats):
    def __init__(self, field=None, weighted='0', Extension=None, Counts=None, NumericInfo=None, DiscrStats=None, ContStats=None, Anova=None):
        super(UnivariateStats, self).__init__(field, weighted, Extension, Counts, NumericInfo, DiscrStats, ContStats, Anova, )

    #
    # XMLBehaviors
    #
supermod.UnivariateStats.subclass = UnivariateStats
# end class UnivariateStats


class Counts(supermod.Counts):
    def __init__(self, totalFreq=None, missingFreq=None, invalidFreq=None, cardinality=None, Extension=None):
        super(Counts, self).__init__(totalFreq, missingFreq, invalidFreq, cardinality, Extension, )

    #
    # XMLBehaviors
    #
supermod.Counts.subclass = Counts
# end class Counts


class NumericInfo(supermod.NumericInfo):
    def __init__(self, minimum=None, maximum=None, mean=None, standardDeviation=None, median=None, interQuartileRange=None, Extension=None, Quantile=None):
        super(NumericInfo, self).__init__(minimum, maximum, mean, standardDeviation, median, interQuartileRange, Extension, Quantile, )

    #
    # XMLBehaviors
    #
supermod.NumericInfo.subclass = NumericInfo
# end class NumericInfo


class Quantile(supermod.Quantile):
    def __init__(self, quantileLimit=None, quantileValue=None, Extension=None):
        super(Quantile, self).__init__(quantileLimit, quantileValue, Extension, )

    #
    # XMLBehaviors
    #
supermod.Quantile.subclass = Quantile
# end class Quantile


class DiscrStats(supermod.DiscrStats):
    def __init__(self, modalValue=None, Extension=None, Array=None):
        super(DiscrStats, self).__init__(modalValue, Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.DiscrStats.subclass = DiscrStats
# end class DiscrStats


class ContStats(supermod.ContStats):
    def __init__(self, totalValuesSum=None, totalSquaresSum=None, Extension=None, Interval=None, NUM_ARRAY=None):
        super(ContStats, self).__init__(totalValuesSum, totalSquaresSum, Extension, Interval, NUM_ARRAY, )

    #
    # XMLBehaviors
    #
supermod.ContStats.subclass = ContStats
# end class ContStats


class MultivariateStats(supermod.MultivariateStats):
    def __init__(self, targetCategory=None, Extension=None, MultivariateStat=None):
        super(MultivariateStats, self).__init__(targetCategory, Extension, MultivariateStat, )

    #
    # XMLBehaviors
    #
supermod.MultivariateStats.subclass = MultivariateStats
# end class MultivariateStats


class MultivariateStat(supermod.MultivariateStat):
    def __init__(self, name=None, category=None, exponent='1', isIntercept=False, importance=None, stdError=None, tValue=None, chiSquareValue=None, fStatistic=None, dF=None, pValueAlpha=None, pValueInitial=None, pValueFinal=None, confidenceLevel='0.95', confidenceLowerBound=None, confidenceUpperBound=None, Extension=None):
        super(MultivariateStat, self).__init__(name, category, exponent, isIntercept, importance, stdError, tValue, chiSquareValue, fStatistic, dF, pValueAlpha, pValueInitial, pValueFinal, confidenceLevel, confidenceLowerBound, confidenceUpperBound, Extension, )

    #
    # XMLBehaviors
    #
supermod.MultivariateStat.subclass = MultivariateStat
# end class MultivariateStat


class Anova(supermod.Anova):
    def __init__(self, target=None, Extension=None, AnovaRow=None):
        super(Anova, self).__init__(target, Extension, AnovaRow, )

    #
    # XMLBehaviors
    #
supermod.Anova.subclass = Anova
# end class Anova


class AnovaRow(supermod.AnovaRow):
    def __init__(self, type_=None, sumOfSquares=None, degreesOfFreedom=None, meanOfSquares=None, fValue=None, pValue=None, Extension=None):
        super(AnovaRow, self).__init__(type_, sumOfSquares, degreesOfFreedom, meanOfSquares, fValue, pValue, Extension, )

    #
    # XMLBehaviors
    #
supermod.AnovaRow.subclass = AnovaRow
# end class AnovaRow


class Partition(supermod.Partition):
    def __init__(self, name=None, size=None, Extension=None, PartitionFieldStats=None):
        super(Partition, self).__init__(name, size, Extension, PartitionFieldStats, )

    #
    # XMLBehaviors
    #
supermod.Partition.subclass = Partition
# end class Partition


class PartitionFieldStats(supermod.PartitionFieldStats):
    def __init__(self, field=None, weighted='0', Extension=None, Counts=None, NumericInfo=None, Array=None):
        super(PartitionFieldStats, self).__init__(field, weighted, Extension, Counts, NumericInfo, Array, )

    #
    # XMLBehaviors
    #
supermod.PartitionFieldStats.subclass = PartitionFieldStats
# end class PartitionFieldStats


class AnomalyDetectionModel(supermod.AnomalyDetectionModel):
    def __init__(self, modelName=None, algorithmName=None, functionName=None, algorithmType=None, sampleDataSize=None, isScorable=True, MiningSchema=None, Output=None, LocalTransformations=None, ModelVerification=None, AnomalyDetectionModel_member=None, AssociationModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, MeanClusterDistances=None, Extension=None):
        super(AnomalyDetectionModel, self).__init__(modelName, algorithmName, functionName, algorithmType, sampleDataSize, isScorable, MiningSchema, Output, LocalTransformations, ModelVerification, AnomalyDetectionModel_member, AssociationModel, BayesianNetworkModel, BaselineModel, ClusteringModel, GaussianProcessModel, GeneralRegressionModel, MiningModel, NaiveBayesModel, NearestNeighborModel, NeuralNetwork, RegressionModel, RuleSetModel, SequenceModel, Scorecard, SupportVectorMachineModel, TextModel, TimeSeriesModel, TreeModel, MeanClusterDistances, Extension, )

    #
    # XMLBehaviors
    #
supermod.AnomalyDetectionModel.subclass = AnomalyDetectionModel
# end class AnomalyDetectionModel


class MeanClusterDistances(supermod.MeanClusterDistances):
    def __init__(self, Extension=None, Array=None):
        super(MeanClusterDistances, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.MeanClusterDistances.subclass = MeanClusterDistances
# end class MeanClusterDistances


class SequenceModel(supermod.SequenceModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, numberOfTransactions=None, maxNumberOfItemsPerTransaction=None, avgNumberOfItemsPerTransaction=None, numberOfTransactionGroups=None, maxNumberOfTAsPerTAGroup=None, avgNumberOfTAsPerTAGroup=None, isScorable=True, MiningSchema=None, ModelStats=None, LocalTransformations=None, Constraints=None, Item=None, Itemset=None, SetPredicate=None, Sequence=None, SequenceRule=None, Extension=None):
        super(SequenceModel, self).__init__(modelName, functionName, algorithmName, numberOfTransactions, maxNumberOfItemsPerTransaction, avgNumberOfItemsPerTransaction, numberOfTransactionGroups, maxNumberOfTAsPerTAGroup, avgNumberOfTAsPerTAGroup, isScorable, MiningSchema, ModelStats, LocalTransformations, Constraints, Item, Itemset, SetPredicate, Sequence, SequenceRule, Extension, )

    #
    # XMLBehaviors
    #
supermod.SequenceModel.subclass = SequenceModel
# end class SequenceModel


class Constraints(supermod.Constraints):
    def __init__(self, minimumNumberOfItems=1, maximumNumberOfItems=None, minimumNumberOfAntecedentItems=1, maximumNumberOfAntecedentItems=None, minimumNumberOfConsequentItems=1, maximumNumberOfConsequentItems=None, minimumSupport='0', minimumConfidence='0', minimumLift='0', minimumTotalSequenceTime='0', maximumTotalSequenceTime=None, minimumItemsetSeparationTime='0', maximumItemsetSeparationTime=None, minimumAntConsSeparationTime='0', maximumAntConsSeparationTime=None, Extension=None):
        super(Constraints, self).__init__(minimumNumberOfItems, maximumNumberOfItems, minimumNumberOfAntecedentItems, maximumNumberOfAntecedentItems, minimumNumberOfConsequentItems, maximumNumberOfConsequentItems, minimumSupport, minimumConfidence, minimumLift, minimumTotalSequenceTime, maximumTotalSequenceTime, minimumItemsetSeparationTime, maximumItemsetSeparationTime, minimumAntConsSeparationTime, maximumAntConsSeparationTime, Extension, )

    #
    # XMLBehaviors
    #
supermod.Constraints.subclass = Constraints
# end class Constraints


class SetPredicate(supermod.SetPredicate):
    def __init__(self, id=None, field=None, operator=None, Extension=None, Array=None):
        super(SetPredicate, self).__init__(id, field, operator, Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.SetPredicate.subclass = SetPredicate
# end class SetPredicate


class Delimiter(supermod.Delimiter):
    def __init__(self, delimiter=None, gap=None, Extension=None):
        super(Delimiter, self).__init__(delimiter, gap, Extension, )

    #
    # XMLBehaviors
    #
supermod.Delimiter.subclass = Delimiter
# end class Delimiter


class Time(supermod.Time):
    def __init__(self, min=None, max=None, mean=None, standardDeviation=None, Extension=None):
        super(Time, self).__init__(min, max, mean, standardDeviation, Extension, )

    #
    # XMLBehaviors
    #
supermod.Time.subclass = Time
# end class Time


class Sequence(supermod.Sequence):
    def __init__(self, id=None, numberOfSets=None, occurrence=None, support=None, Extension=None, Delimiter=None, SetReference=None, Time=None):
        super(Sequence, self).__init__(id, numberOfSets, occurrence, support, Extension, Delimiter, SetReference, Time, )

    #
    # XMLBehaviors
    #
supermod.Sequence.subclass = Sequence
# end class Sequence


class SetReference(supermod.SetReference):
    def __init__(self, setId=None, Extension=None):
        super(SetReference, self).__init__(setId, Extension, )

    #
    # XMLBehaviors
    #
supermod.SetReference.subclass = SetReference
# end class SetReference


class SequenceRule(supermod.SequenceRule):
    def __init__(self, id=None, numberOfSets=None, occurrence=None, support=None, confidence=None, lift=None, Extension=None, AntecedentSequence=None, Delimiter=None, ConsequentSequence=None, Time=None):
        super(SequenceRule, self).__init__(id, numberOfSets, occurrence, support, confidence, lift, Extension, AntecedentSequence, Delimiter, ConsequentSequence, Time, )

    #
    # XMLBehaviors
    #
supermod.SequenceRule.subclass = SequenceRule
# end class SequenceRule


class SequenceReference(supermod.SequenceReference):
    def __init__(self, seqId=None, Extension=None):
        super(SequenceReference, self).__init__(seqId, Extension, )

    #
    # XMLBehaviors
    #
supermod.SequenceReference.subclass = SequenceReference
# end class SequenceReference


class AntecedentSequence(supermod.AntecedentSequence):
    def __init__(self, Extension=None, SequenceReference=None, Time=None):
        super(AntecedentSequence, self).__init__(Extension, SequenceReference, Time, )

    #
    # XMLBehaviors
    #
supermod.AntecedentSequence.subclass = AntecedentSequence
# end class AntecedentSequence


class ConsequentSequence(supermod.ConsequentSequence):
    def __init__(self, Extension=None, SequenceReference=None, Time=None):
        super(ConsequentSequence, self).__init__(Extension, SequenceReference, Time, )

    #
    # XMLBehaviors
    #
supermod.ConsequentSequence.subclass = ConsequentSequence
# end class ConsequentSequence


class ModelVerification(supermod.ModelVerification):
    def __init__(self, recordCount=None, fieldCount=None, Extension=None, VerificationFields=None, InlineTable=None):
        super(ModelVerification, self).__init__(recordCount, fieldCount, Extension, VerificationFields, InlineTable, )

    #
    # XMLBehaviors
    #
supermod.ModelVerification.subclass = ModelVerification
# end class ModelVerification


class VerificationFields(supermod.VerificationFields):
    def __init__(self, Extension=None, VerificationField=None):
        super(VerificationFields, self).__init__(Extension, VerificationField, )

    #
    # XMLBehaviors
    #
supermod.VerificationFields.subclass = VerificationFields
# end class VerificationFields


class VerificationField(supermod.VerificationField):
    def __init__(self, field=None, column=None, precision=1E-6, zeroThreshold=1E-16, Extension=None):
        super(VerificationField, self).__init__(field, column, precision, zeroThreshold, Extension, )

    #
    # XMLBehaviors
    #
supermod.VerificationField.subclass = VerificationField
# end class VerificationField


class GeneralRegressionModel(supermod.GeneralRegressionModel):
    def __init__(self, targetVariableName=None, modelType=None, modelName=None, functionName=None, algorithmName=None, targetReferenceCategory=None, cumulativeLink=None, linkFunction=None, linkParameter=None, trialsVariable=None, trialsValue=None, distribution=None, distParameter=None, offsetVariable=None, offsetValue=None, modelDF=None, endTimeVariable=None, startTimeVariable=None, subjectIDVariable=None, statusVariable=None, baselineStrataVariable=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, ParameterList=None, FactorList=None, CovariateList=None, PPMatrix=None, PCovMatrix=None, ParamMatrix=None, EventValues=None, BaseCumHazardTables=None, ModelVerification=None, Extension=None):
        super(GeneralRegressionModel, self).__init__(targetVariableName, modelType, modelName, functionName, algorithmName, targetReferenceCategory, cumulativeLink, linkFunction, linkParameter, trialsVariable, trialsValue, distribution, distParameter, offsetVariable, offsetValue, modelDF, endTimeVariable, startTimeVariable, subjectIDVariable, statusVariable, baselineStrataVariable, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, ParameterList, FactorList, CovariateList, PPMatrix, PCovMatrix, ParamMatrix, EventValues, BaseCumHazardTables, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.GeneralRegressionModel.subclass = GeneralRegressionModel
# end class GeneralRegressionModel


class ParameterList(supermod.ParameterList):
    def __init__(self, Extension=None, Parameter=None):
        super(ParameterList, self).__init__(Extension, Parameter, )

    #
    # XMLBehaviors
    #
supermod.ParameterList.subclass = ParameterList
# end class ParameterList


class Parameter(supermod.Parameter):
    def __init__(self, name=None, label=None, referencePoint='0', Extension=None):
        super(Parameter, self).__init__(name, label, referencePoint, Extension, )

    #
    # XMLBehaviors
    #
supermod.Parameter.subclass = Parameter
# end class Parameter


class FactorList(supermod.FactorList):
    def __init__(self, Extension=None, Predictor=None):
        super(FactorList, self).__init__(Extension, Predictor, )

    #
    # XMLBehaviors
    #
supermod.FactorList.subclass = FactorList
# end class FactorList


class CovariateList(supermod.CovariateList):
    def __init__(self, Extension=None, Predictor=None):
        super(CovariateList, self).__init__(Extension, Predictor, )

    #
    # XMLBehaviors
    #
supermod.CovariateList.subclass = CovariateList
# end class CovariateList


class Predictor(supermod.Predictor):
    def __init__(self, name=None, contrastMatrixType=None, Extension=None, Categories=None, Matrix=None):
        super(Predictor, self).__init__(name, contrastMatrixType, Extension, Categories, Matrix, )

    #
    # XMLBehaviors
    #
supermod.Predictor.subclass = Predictor
# end class Predictor


class Categories(supermod.Categories):
    def __init__(self, Extension=None, Category=None):
        super(Categories, self).__init__(Extension, Category, )

    #
    # XMLBehaviors
    #
supermod.Categories.subclass = Categories
# end class Categories


class Category(supermod.Category):
    def __init__(self, value=None, Extension=None):
        super(Category, self).__init__(value, Extension, )

    #
    # XMLBehaviors
    #
supermod.Category.subclass = Category
# end class Category


class PPMatrix(supermod.PPMatrix):
    def __init__(self, Extension=None, PPCell=None):
        super(PPMatrix, self).__init__(Extension, PPCell, )

    #
    # XMLBehaviors
    #
supermod.PPMatrix.subclass = PPMatrix
# end class PPMatrix


class PPCell(supermod.PPCell):
    def __init__(self, value=None, predictorName=None, parameterName=None, targetCategory=None, Extension=None):
        super(PPCell, self).__init__(value, predictorName, parameterName, targetCategory, Extension, )

    #
    # XMLBehaviors
    #
supermod.PPCell.subclass = PPCell
# end class PPCell


class PCovMatrix(supermod.PCovMatrix):
    def __init__(self, type_=None, Extension=None, PCovCell=None):
        super(PCovMatrix, self).__init__(type_, Extension, PCovCell, )

    #
    # XMLBehaviors
    #
supermod.PCovMatrix.subclass = PCovMatrix
# end class PCovMatrix


class PCovCell(supermod.PCovCell):
    def __init__(self, pRow=None, pCol=None, tRow=None, tCol=None, value=None, targetCategory=None, Extension=None):
        super(PCovCell, self).__init__(pRow, pCol, tRow, tCol, value, targetCategory, Extension, )

    #
    # XMLBehaviors
    #
supermod.PCovCell.subclass = PCovCell
# end class PCovCell


class ParamMatrix(supermod.ParamMatrix):
    def __init__(self, Extension=None, PCell=None):
        super(ParamMatrix, self).__init__(Extension, PCell, )

    #
    # XMLBehaviors
    #
supermod.ParamMatrix.subclass = ParamMatrix
# end class ParamMatrix


class PCell(supermod.PCell):
    def __init__(self, targetCategory=None, parameterName=None, beta=None, df=None, Extension=None):
        super(PCell, self).__init__(targetCategory, parameterName, beta, df, Extension, )

    #
    # XMLBehaviors
    #
supermod.PCell.subclass = PCell
# end class PCell


class BaseCumHazardTables(supermod.BaseCumHazardTables):
    def __init__(self, maxTime=None, Extension=None, BaselineStratum=None, BaselineCell=None):
        super(BaseCumHazardTables, self).__init__(maxTime, Extension, BaselineStratum, BaselineCell, )

    #
    # XMLBehaviors
    #
supermod.BaseCumHazardTables.subclass = BaseCumHazardTables
# end class BaseCumHazardTables


class BaselineStratum(supermod.BaselineStratum):
    def __init__(self, value=None, label=None, maxTime=None, Extension=None, BaselineCell=None):
        super(BaselineStratum, self).__init__(value, label, maxTime, Extension, BaselineCell, )

    #
    # XMLBehaviors
    #
supermod.BaselineStratum.subclass = BaselineStratum
# end class BaselineStratum


class BaselineCell(supermod.BaselineCell):
    def __init__(self, time=None, cumHazard=None, Extension=None):
        super(BaselineCell, self).__init__(time, cumHazard, Extension, )

    #
    # XMLBehaviors
    #
supermod.BaselineCell.subclass = BaselineCell
# end class BaselineCell


class EventValues(supermod.EventValues):
    def __init__(self, Extension=None, Value=None, Interval=None):
        super(EventValues, self).__init__(Extension, Value, Interval, )

    #
    # XMLBehaviors
    #
supermod.EventValues.subclass = EventValues
# end class EventValues


class NearestNeighborModel(supermod.NearestNeighborModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, numberOfNeighbors=None, continuousScoringMethod='average', categoricalScoringMethod='majorityVote', instanceIdVariable=None, threshold='0.001', isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, TrainingInstances=None, ComparisonMeasure=None, KNNInputs=None, ModelVerification=None, Extension=None):
        super(NearestNeighborModel, self).__init__(modelName, functionName, algorithmName, numberOfNeighbors, continuousScoringMethod, categoricalScoringMethod, instanceIdVariable, threshold, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, TrainingInstances, ComparisonMeasure, KNNInputs, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.NearestNeighborModel.subclass = NearestNeighborModel
# end class NearestNeighborModel


class TrainingInstances(supermod.TrainingInstances):
    def __init__(self, isTransformed=False, recordCount=None, fieldCount=None, Extension=None, InstanceFields=None, TableLocator=None, InlineTable=None):
        super(TrainingInstances, self).__init__(isTransformed, recordCount, fieldCount, Extension, InstanceFields, TableLocator, InlineTable, )

    #
    # XMLBehaviors
    #
supermod.TrainingInstances.subclass = TrainingInstances
# end class TrainingInstances


class InstanceFields(supermod.InstanceFields):
    def __init__(self, Extension=None, InstanceField=None):
        super(InstanceFields, self).__init__(Extension, InstanceField, )

    #
    # XMLBehaviors
    #
supermod.InstanceFields.subclass = InstanceFields
# end class InstanceFields


class InstanceField(supermod.InstanceField):
    def __init__(self, field=None, column=None, Extension=None):
        super(InstanceField, self).__init__(field, column, Extension, )

    #
    # XMLBehaviors
    #
supermod.InstanceField.subclass = InstanceField
# end class InstanceField


class KNNInputs(supermod.KNNInputs):
    def __init__(self, Extension=None, KNNInput=None):
        super(KNNInputs, self).__init__(Extension, KNNInput, )

    #
    # XMLBehaviors
    #
supermod.KNNInputs.subclass = KNNInputs
# end class KNNInputs


class KNNInput(supermod.KNNInput):
    def __init__(self, field=None, fieldWeight='1', compareFunction=None, Extension=None):
        super(KNNInput, self).__init__(field, fieldWeight, compareFunction, Extension, )

    #
    # XMLBehaviors
    #
supermod.KNNInput.subclass = KNNInput
# end class KNNInput


class TransformationDictionary(supermod.TransformationDictionary):
    def __init__(self, Extension=None, DefineFunction=None, DerivedField=None):
        super(TransformationDictionary, self).__init__(Extension, DefineFunction, DerivedField, )

    #
    # XMLBehaviors
    #
supermod.TransformationDictionary.subclass = TransformationDictionary
# end class TransformationDictionary


class LocalTransformations(supermod.LocalTransformations):
    def __init__(self, Extension=None, DerivedField=None):
        super(LocalTransformations, self).__init__(Extension, DerivedField, )

    #
    # XMLBehaviors
    #
supermod.LocalTransformations.subclass = LocalTransformations
# end class LocalTransformations


class DerivedField(supermod.DerivedField):
    def __init__(self, name=None, displayName=None, optype=None, dataType=None, Extension=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None, Value=None):
        super(DerivedField, self).__init__(name, displayName, optype, dataType, Extension, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, Value, )

    #
    # XMLBehaviors
    #
supermod.DerivedField.subclass = DerivedField
# end class DerivedField


class Constant(supermod.Constant):
    def __init__(self, dataType=None, missing=False, valueOf_=None):
        super(Constant, self).__init__(dataType, missing, valueOf_, )

    #
    # XMLBehaviors
    #
supermod.Constant.subclass = Constant
# end class Constant


class FieldRef(supermod.FieldRef):
    def __init__(self, field=None, mapMissingTo=None, Extension=None):
        super(FieldRef, self).__init__(field, mapMissingTo, Extension, )

    #
    # XMLBehaviors
    #
supermod.FieldRef.subclass = FieldRef
# end class FieldRef


class NormContinuous(supermod.NormContinuous):
    def __init__(self, mapMissingTo=None, field=None, outliers='asIs', Extension=None, LinearNorm=None):
        super(NormContinuous, self).__init__(mapMissingTo, field, outliers, Extension, LinearNorm, )

    #
    # XMLBehaviors
    #
supermod.NormContinuous.subclass = NormContinuous
# end class NormContinuous


class LinearNorm(supermod.LinearNorm):
    def __init__(self, orig=None, norm=None, Extension=None):
        super(LinearNorm, self).__init__(orig, norm, Extension, )

    #
    # XMLBehaviors
    #
supermod.LinearNorm.subclass = LinearNorm
# end class LinearNorm


class NormDiscrete(supermod.NormDiscrete):
    def __init__(self, field=None, value=None, mapMissingTo=None, Extension=None):
        super(NormDiscrete, self).__init__(field, value, mapMissingTo, Extension, )

    #
    # XMLBehaviors
    #
supermod.NormDiscrete.subclass = NormDiscrete
# end class NormDiscrete


class Discretize(supermod.Discretize):
    def __init__(self, field=None, mapMissingTo=None, defaultValue=None, dataType=None, Extension=None, DiscretizeBin=None):
        super(Discretize, self).__init__(field, mapMissingTo, defaultValue, dataType, Extension, DiscretizeBin, )

    #
    # XMLBehaviors
    #
supermod.Discretize.subclass = Discretize
# end class Discretize


class DiscretizeBin(supermod.DiscretizeBin):
    def __init__(self, binValue=None, Extension=None, Interval=None):
        super(DiscretizeBin, self).__init__(binValue, Extension, Interval, )

    #
    # XMLBehaviors
    #
supermod.DiscretizeBin.subclass = DiscretizeBin
# end class DiscretizeBin


class MapValues(supermod.MapValues):
    def __init__(self, mapMissingTo=None, defaultValue=None, outputColumn=None, dataType=None, Extension=None, FieldColumnPair=None, TableLocator=None, InlineTable=None):
        super(MapValues, self).__init__(mapMissingTo, defaultValue, outputColumn, dataType, Extension, FieldColumnPair, TableLocator, InlineTable, )

    #
    # XMLBehaviors
    #
supermod.MapValues.subclass = MapValues
# end class MapValues


class FieldColumnPair(supermod.FieldColumnPair):
    def __init__(self, field=None, column=None, Extension=None):
        super(FieldColumnPair, self).__init__(field, column, Extension, )

    #
    # XMLBehaviors
    #
supermod.FieldColumnPair.subclass = FieldColumnPair
# end class FieldColumnPair


class TextIndex(supermod.TextIndex):
    def __init__(self, textField=None, localTermWeights='termFrequency', isCaseSensitive=False, maxLevenshteinDistance=0, countHits='allHits', wordSeparatorCharacterRE='\\s+', tokenize=True, Extension=None, TextIndexNormalization=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex_member=None, Aggregate=None, Lag=None):
        super(TextIndex, self).__init__(textField, localTermWeights, isCaseSensitive, maxLevenshteinDistance, countHits, wordSeparatorCharacterRE, tokenize, Extension, TextIndexNormalization, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex_member, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.TextIndex.subclass = TextIndex
# end class TextIndex


class TextIndexNormalization(supermod.TextIndexNormalization):
    def __init__(self, inField='string', outField='stem', regexField='regex', recursive=False, isCaseSensitive=None, maxLevenshteinDistance=None, wordSeparatorCharacterRE=None, tokenize=None, Extension=None, TableLocator=None, InlineTable=None):
        super(TextIndexNormalization, self).__init__(inField, outField, regexField, recursive, isCaseSensitive, maxLevenshteinDistance, wordSeparatorCharacterRE, tokenize, Extension, TableLocator, InlineTable, )

    #
    # XMLBehaviors
    #
supermod.TextIndexNormalization.subclass = TextIndexNormalization
# end class TextIndexNormalization


class Aggregate(supermod.Aggregate):
    def __init__(self, field=None, function=None, groupField=None, sqlWhere=None, Extension=None):
        super(Aggregate, self).__init__(field, function, groupField, sqlWhere, Extension, )

    #
    # XMLBehaviors
    #
supermod.Aggregate.subclass = Aggregate
# end class Aggregate


class Lag(supermod.Lag):
    def __init__(self, field=None, n=1, aggregate='none', Extension=None, BlockIndicator=None):
        super(Lag, self).__init__(field, n, aggregate, Extension, BlockIndicator, )

    #
    # XMLBehaviors
    #
supermod.Lag.subclass = Lag
# end class Lag


class BlockIndicator(supermod.BlockIndicator):
    def __init__(self, field=None, Extension=None):
        super(BlockIndicator, self).__init__(field, Extension, )

    #
    # XMLBehaviors
    #
supermod.BlockIndicator.subclass = BlockIndicator
# end class BlockIndicator


class TimeSeriesModel(supermod.TimeSeriesModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, bestFit=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, LocalTransformations=None, TimeSeries=None, SpectralAnalysis=None, ARIMA=None, ExponentialSmoothing=None, SeasonalTrendDecomposition=None, StateSpaceModel=None, GARCH=None, ModelVerification=None, Extension=None):
        super(TimeSeriesModel, self).__init__(modelName, functionName, algorithmName, bestFit, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, LocalTransformations, TimeSeries, SpectralAnalysis, ARIMA, ExponentialSmoothing, SeasonalTrendDecomposition, StateSpaceModel, GARCH, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.TimeSeriesModel.subclass = TimeSeriesModel
# end class TimeSeriesModel


class TimeSeries(supermod.TimeSeries):
    def __init__(self, usage='original', startTime=None, endTime=None, interpolationMethod='none', field=None, TimeAnchor=None, TimeValue=None):
        super(TimeSeries, self).__init__(usage, startTime, endTime, interpolationMethod, field, TimeAnchor, TimeValue, )

    #
    # XMLBehaviors
    #
supermod.TimeSeries.subclass = TimeSeries
# end class TimeSeries


class TimeValue(supermod.TimeValue):
    def __init__(self, index=None, time=None, value=None, standardError=None, Timestamp=None):
        super(TimeValue, self).__init__(index, time, value, standardError, Timestamp, )

    #
    # XMLBehaviors
    #
supermod.TimeValue.subclass = TimeValue
# end class TimeValue


class TimeAnchor(supermod.TimeAnchor):
    def __init__(self, type_=None, offset=None, stepsize=None, displayName=None, TimeCycle=None, TimeException=None):
        super(TimeAnchor, self).__init__(type_, offset, stepsize, displayName, TimeCycle, TimeException, )

    #
    # XMLBehaviors
    #
supermod.TimeAnchor.subclass = TimeAnchor
# end class TimeAnchor


class TimeCycle(supermod.TimeCycle):
    def __init__(self, length=None, type_=None, displayName=None, Array=None):
        super(TimeCycle, self).__init__(length, type_, displayName, Array, )

    #
    # XMLBehaviors
    #
supermod.TimeCycle.subclass = TimeCycle
# end class TimeCycle


class TimeException(supermod.TimeException):
    def __init__(self, type_=None, count=None, Array=None):
        super(TimeException, self).__init__(type_, count, Array, )

    #
    # XMLBehaviors
    #
supermod.TimeException.subclass = TimeException
# end class TimeException


class ExponentialSmoothing(supermod.ExponentialSmoothing):
    def __init__(self, RMSE=None, transformation='none', Level=None, Trend_ExpoSmooth=None, Seasonality_ExpoSmooth=None, TimeValue=None):
        super(ExponentialSmoothing, self).__init__(RMSE, transformation, Level, Trend_ExpoSmooth, Seasonality_ExpoSmooth, TimeValue, )

    #
    # XMLBehaviors
    #
supermod.ExponentialSmoothing.subclass = ExponentialSmoothing
# end class ExponentialSmoothing


class Level(supermod.Level):
    def __init__(self, alpha=None, smoothedValue=None):
        super(Level, self).__init__(alpha, smoothedValue, )

    #
    # XMLBehaviors
    #
supermod.Level.subclass = Level
# end class Level


class Trend_ExpoSmooth(supermod.Trend_ExpoSmooth):
    def __init__(self, trend='additive', gamma=None, phi='1', smoothedValue=None, Array=None):
        super(Trend_ExpoSmooth, self).__init__(trend, gamma, phi, smoothedValue, Array, )

    #
    # XMLBehaviors
    #
supermod.Trend_ExpoSmooth.subclass = Trend_ExpoSmooth
# end class Trend_ExpoSmooth


class Seasonality_ExpoSmooth(supermod.Seasonality_ExpoSmooth):
    def __init__(self, type_=None, period=None, unit=None, phase=None, delta=None, Array=None):
        super(Seasonality_ExpoSmooth, self).__init__(type_, period, unit, phase, delta, Array, )

    #
    # XMLBehaviors
    #
supermod.Seasonality_ExpoSmooth.subclass = Seasonality_ExpoSmooth
# end class Seasonality_ExpoSmooth


class ARIMA(supermod.ARIMA):
    def __init__(self, RMSE=None, transformation='none', constantTerm='0', predictionMethod='conditionalLeastSquares', Extension=None, NonseasonalComponent=None, SeasonalComponent=None, DynamicRegressor=None, MaximumLikelihoodStat=None, OutlierEffect=None):
        super(ARIMA, self).__init__(RMSE, transformation, constantTerm, predictionMethod, Extension, NonseasonalComponent, SeasonalComponent, DynamicRegressor, MaximumLikelihoodStat, OutlierEffect, )

    #
    # XMLBehaviors
    #
supermod.ARIMA.subclass = ARIMA
# end class ARIMA


class NonseasonalComponent(supermod.NonseasonalComponent):
    def __init__(self, p=0, d=0, q=0, Extension=None, AR=None, MA=None):
        super(NonseasonalComponent, self).__init__(p, d, q, Extension, AR, MA, )

    #
    # XMLBehaviors
    #
supermod.NonseasonalComponent.subclass = NonseasonalComponent
# end class NonseasonalComponent


class SeasonalComponent(supermod.SeasonalComponent):
    def __init__(self, P=0, D=0, Q=0, period=None, Extension=None, AR=None, MA=None):
        super(SeasonalComponent, self).__init__(P, D, Q, period, Extension, AR, MA, )

    #
    # XMLBehaviors
    #
supermod.SeasonalComponent.subclass = SeasonalComponent
# end class SeasonalComponent


class AR(supermod.AR):
    def __init__(self, Extension=None, Array=None):
        super(AR, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.AR.subclass = AR
# end class AR


class MA(supermod.MA):
    def __init__(self, Extension=None, MACoefficients=None, Residuals=None):
        super(MA, self).__init__(Extension, MACoefficients, Residuals, )

    #
    # XMLBehaviors
    #
supermod.MA.subclass = MA
# end class MA


class MACoefficients(supermod.MACoefficients):
    def __init__(self, Extension=None, Array=None):
        super(MACoefficients, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.MACoefficients.subclass = MACoefficients
# end class MACoefficients


class Residuals(supermod.Residuals):
    def __init__(self, Extension=None, Array=None):
        super(Residuals, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.Residuals.subclass = Residuals
# end class Residuals


class DynamicRegressor(supermod.DynamicRegressor):
    def __init__(self, field=None, transformation='none', delay='0', futureValuesMethod='constant', targetField=None, Extension=None, Numerator=None, Denominator=None, RegressorValues=None):
        super(DynamicRegressor, self).__init__(field, transformation, delay, futureValuesMethod, targetField, Extension, Numerator, Denominator, RegressorValues, )

    #
    # XMLBehaviors
    #
supermod.DynamicRegressor.subclass = DynamicRegressor
# end class DynamicRegressor


class Numerator(supermod.Numerator):
    def __init__(self, Extension=None, NonseasonalFactor=None, SeasonalFactor=None):
        super(Numerator, self).__init__(Extension, NonseasonalFactor, SeasonalFactor, )

    #
    # XMLBehaviors
    #
supermod.Numerator.subclass = Numerator
# end class Numerator


class Denominator(supermod.Denominator):
    def __init__(self, Extension=None, NonseasonalFactor=None, SeasonalFactor=None):
        super(Denominator, self).__init__(Extension, NonseasonalFactor, SeasonalFactor, )

    #
    # XMLBehaviors
    #
supermod.Denominator.subclass = Denominator
# end class Denominator


class SeasonalFactor(supermod.SeasonalFactor):
    def __init__(self, difference='0', maximumOrder=None, Extension=None, Array=None):
        super(SeasonalFactor, self).__init__(difference, maximumOrder, Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.SeasonalFactor.subclass = SeasonalFactor
# end class SeasonalFactor


class NonseasonalFactor(supermod.NonseasonalFactor):
    def __init__(self, difference='0', maximumOrder=None, Extension=None, Array=None):
        super(NonseasonalFactor, self).__init__(difference, maximumOrder, Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.NonseasonalFactor.subclass = NonseasonalFactor
# end class NonseasonalFactor


class RegressorValues(supermod.RegressorValues):
    def __init__(self, Extension=None, TimeSeries=None, TrendCoefficients=None, TransferFunctionValues=None):
        super(RegressorValues, self).__init__(Extension, TimeSeries, TrendCoefficients, TransferFunctionValues, )

    #
    # XMLBehaviors
    #
supermod.RegressorValues.subclass = RegressorValues
# end class RegressorValues


class TrendCoefficients(supermod.TrendCoefficients):
    def __init__(self, Extension=None, REAL_SparseArray=None):
        super(TrendCoefficients, self).__init__(Extension, REAL_SparseArray, )

    #
    # XMLBehaviors
    #
supermod.TrendCoefficients.subclass = TrendCoefficients
# end class TrendCoefficients


class TransferFunctionValues(supermod.TransferFunctionValues):
    def __init__(self, Array=None):
        super(TransferFunctionValues, self).__init__(Array, )

    #
    # XMLBehaviors
    #
supermod.TransferFunctionValues.subclass = TransferFunctionValues
# end class TransferFunctionValues


class MaximumLikelihoodStat(supermod.MaximumLikelihoodStat):
    def __init__(self, method=None, periodDeficit='0', KalmanState=None, ThetaRecursionState=None):
        super(MaximumLikelihoodStat, self).__init__(method, periodDeficit, KalmanState, ThetaRecursionState, )

    #
    # XMLBehaviors
    #
supermod.MaximumLikelihoodStat.subclass = MaximumLikelihoodStat
# end class MaximumLikelihoodStat


class KalmanState(supermod.KalmanState):
    def __init__(self, FinalOmega=None, FinalStateVector=None, HVector=None):
        super(KalmanState, self).__init__(FinalOmega, FinalStateVector, HVector, )

    #
    # XMLBehaviors
    #
supermod.KalmanState.subclass = KalmanState
# end class KalmanState


class FinalOmega(supermod.FinalOmega):
    def __init__(self, Matrix=None):
        super(FinalOmega, self).__init__(Matrix, )

    #
    # XMLBehaviors
    #
supermod.FinalOmega.subclass = FinalOmega
# end class FinalOmega


class FinalStateVector(supermod.FinalStateVector):
    def __init__(self, Array=None):
        super(FinalStateVector, self).__init__(Array, )

    #
    # XMLBehaviors
    #
supermod.FinalStateVector.subclass = FinalStateVector
# end class FinalStateVector


class HVector(supermod.HVector):
    def __init__(self, Array=None):
        super(HVector, self).__init__(Array, )

    #
    # XMLBehaviors
    #
supermod.HVector.subclass = HVector
# end class HVector


class ThetaRecursionState(supermod.ThetaRecursionState):
    def __init__(self, FinalNoise=None, FinalPredictedNoise=None, FinalTheta=None, FinalNu=None):
        super(ThetaRecursionState, self).__init__(FinalNoise, FinalPredictedNoise, FinalTheta, FinalNu, )

    #
    # XMLBehaviors
    #
supermod.ThetaRecursionState.subclass = ThetaRecursionState
# end class ThetaRecursionState


class FinalNoise(supermod.FinalNoise):
    def __init__(self, Array=None):
        super(FinalNoise, self).__init__(Array, )

    #
    # XMLBehaviors
    #
supermod.FinalNoise.subclass = FinalNoise
# end class FinalNoise


class FinalPredictedNoise(supermod.FinalPredictedNoise):
    def __init__(self, Array=None):
        super(FinalPredictedNoise, self).__init__(Array, )

    #
    # XMLBehaviors
    #
supermod.FinalPredictedNoise.subclass = FinalPredictedNoise
# end class FinalPredictedNoise


class FinalTheta(supermod.FinalTheta):
    def __init__(self, Theta=None):
        super(FinalTheta, self).__init__(Theta, )

    #
    # XMLBehaviors
    #
supermod.FinalTheta.subclass = FinalTheta
# end class FinalTheta


class Theta(supermod.Theta):
    def __init__(self, i=None, j=None, theta=None):
        super(Theta, self).__init__(i, j, theta, )

    #
    # XMLBehaviors
    #
supermod.Theta.subclass = Theta
# end class Theta


class FinalNu(supermod.FinalNu):
    def __init__(self, Array=None):
        super(FinalNu, self).__init__(Array, )

    #
    # XMLBehaviors
    #
supermod.FinalNu.subclass = FinalNu
# end class FinalNu


class OutlierEffect(supermod.OutlierEffect):
    def __init__(self, type_=None, startTime=None, magnitude=None, dampingCoefficient=None, Extension=None):
        super(OutlierEffect, self).__init__(type_, startTime, magnitude, dampingCoefficient, Extension, )

    #
    # XMLBehaviors
    #
supermod.OutlierEffect.subclass = OutlierEffect
# end class OutlierEffect


class GARCH(supermod.GARCH):
    def __init__(self, Extension=None, ARMAPart=None, GARCHPart=None):
        super(GARCH, self).__init__(Extension, ARMAPart, GARCHPart, )

    #
    # XMLBehaviors
    #
supermod.GARCH.subclass = GARCH
# end class GARCH


class ARMAPart(supermod.ARMAPart):
    def __init__(self, constant='0', p=None, q=None, Extension=None, AR=None, MA=None):
        super(ARMAPart, self).__init__(constant, p, q, Extension, AR, MA, )

    #
    # XMLBehaviors
    #
supermod.ARMAPart.subclass = ARMAPart
# end class ARMAPart


class GARCHPart(supermod.GARCHPart):
    def __init__(self, constant='0', gp=None, gq=None, Extension=None, ResidualSquareCoefficients=None, VarianceCoefficients=None):
        super(GARCHPart, self).__init__(constant, gp, gq, Extension, ResidualSquareCoefficients, VarianceCoefficients, )

    #
    # XMLBehaviors
    #
supermod.GARCHPart.subclass = GARCHPart
# end class GARCHPart


class ResidualSquareCoefficients(supermod.ResidualSquareCoefficients):
    def __init__(self, Extension=None, Residuals=None, MACoefficients=None):
        super(ResidualSquareCoefficients, self).__init__(Extension, Residuals, MACoefficients, )

    #
    # XMLBehaviors
    #
supermod.ResidualSquareCoefficients.subclass = ResidualSquareCoefficients
# end class ResidualSquareCoefficients


class VarianceCoefficients(supermod.VarianceCoefficients):
    def __init__(self, Extension=None, PastVariances=None, MACoefficients=None):
        super(VarianceCoefficients, self).__init__(Extension, PastVariances, MACoefficients, )

    #
    # XMLBehaviors
    #
supermod.VarianceCoefficients.subclass = VarianceCoefficients
# end class VarianceCoefficients


class PastVariances(supermod.PastVariances):
    def __init__(self, Extension=None, Array=None):
        super(PastVariances, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.PastVariances.subclass = PastVariances
# end class PastVariances


class StateSpaceModel(supermod.StateSpaceModel):
    def __init__(self, variance=None, period='none', intercept='0', Extension=None, StateVector=None, TransitionMatrix=None, MeasurementMatrix=None, InterceptVector=None, PredictedStateCovarianceMatrix=None, SelectedStateCovarianceMatrix=None, ObservationVarianceMatrix=None, PsiVector=None, DynamicRegressor=None):
        super(StateSpaceModel, self).__init__(variance, period, intercept, Extension, StateVector, TransitionMatrix, MeasurementMatrix, InterceptVector, PredictedStateCovarianceMatrix, SelectedStateCovarianceMatrix, ObservationVarianceMatrix, PsiVector, DynamicRegressor, )

    #
    # XMLBehaviors
    #
supermod.StateSpaceModel.subclass = StateSpaceModel
# end class StateSpaceModel


class StateVector(supermod.StateVector):
    def __init__(self, Extension=None, Array=None):
        super(StateVector, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.StateVector.subclass = StateVector
# end class StateVector


class TransitionMatrix(supermod.TransitionMatrix):
    def __init__(self, Extension=None, Matrix=None):
        super(TransitionMatrix, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.TransitionMatrix.subclass = TransitionMatrix
# end class TransitionMatrix


class MeasurementMatrix(supermod.MeasurementMatrix):
    def __init__(self, Extension=None, Matrix=None):
        super(MeasurementMatrix, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.MeasurementMatrix.subclass = MeasurementMatrix
# end class MeasurementMatrix


class InterceptVector(supermod.InterceptVector):
    def __init__(self, type_='state', Extension=None, Array=None):
        super(InterceptVector, self).__init__(type_, Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.InterceptVector.subclass = InterceptVector
# end class InterceptVector


class PredictedStateCovarianceMatrix(supermod.PredictedStateCovarianceMatrix):
    def __init__(self, Extension=None, Matrix=None):
        super(PredictedStateCovarianceMatrix, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.PredictedStateCovarianceMatrix.subclass = PredictedStateCovarianceMatrix
# end class PredictedStateCovarianceMatrix


class SelectedStateCovarianceMatrix(supermod.SelectedStateCovarianceMatrix):
    def __init__(self, Extension=None, Matrix=None):
        super(SelectedStateCovarianceMatrix, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.SelectedStateCovarianceMatrix.subclass = SelectedStateCovarianceMatrix
# end class SelectedStateCovarianceMatrix


class ObservationVarianceMatrix(supermod.ObservationVarianceMatrix):
    def __init__(self, Extension=None, Matrix=None):
        super(ObservationVarianceMatrix, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.ObservationVarianceMatrix.subclass = ObservationVarianceMatrix
# end class ObservationVarianceMatrix


class PsiVector(supermod.PsiVector):
    def __init__(self, targetField=None, variance=None, Extension=None, Array=None):
        super(PsiVector, self).__init__(targetField, variance, Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.PsiVector.subclass = PsiVector
# end class PsiVector


class SpectralAnalysis(supermod.SpectralAnalysis):
    def __init__(self, Extension=None):
        super(SpectralAnalysis, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.SpectralAnalysis.subclass = SpectralAnalysis
# end class SpectralAnalysis


class SeasonalTrendDecomposition(supermod.SeasonalTrendDecomposition):
    def __init__(self, Extension=None):
        super(SeasonalTrendDecomposition, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.SeasonalTrendDecomposition.subclass = SeasonalTrendDecomposition
# end class SeasonalTrendDecomposition


class NeuralNetwork(supermod.NeuralNetwork):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, activationFunction=None, normalizationMethod='none', threshold='0', width=None, altitude='1.0', numberOfLayers=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, NeuralInputs=None, NeuralLayer=None, NeuralOutputs=None, ModelVerification=None, Extension=None):
        super(NeuralNetwork, self).__init__(modelName, functionName, algorithmName, activationFunction, normalizationMethod, threshold, width, altitude, numberOfLayers, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, NeuralInputs, NeuralLayer, NeuralOutputs, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
    def set_NeuralLayer(self, NeuralLayer, *args):
        self.NeuralLayer = NeuralLayer

        self.numberOfLayers = len(self.NeuralLayer)
    def set_NeuralLayer_wrapper(self, NeuralLayer, *args):
        result = self.set_NeuralLayer(NeuralLayer, *args)
        return result

    def add_NeuralLayer(self, value, *args):
        self.NeuralLayer.append(value)

        self.numberOfLayers = len(self.NeuralLayer)
    def add_NeuralLayer_wrapper(self, value, *args):
        result = self.add_NeuralLayer(value, *args)
        return result

    def insert_NeuralLayer_at(self, index, value, *args):
        self.NeuralLayer.insert(index, value)

        self.numberOfLayers = len(self.NeuralLayer)
    def insert_NeuralLayer_at_wrapper(self, index, value, *args):
        result = self.insert_NeuralLayer_at(index, value, *args)
        return result

supermod.NeuralNetwork.subclass = NeuralNetwork
# end class NeuralNetwork


class NeuralInputs(supermod.NeuralInputs):
    def __init__(self, numberOfInputs=None, Extension=None, NeuralInput=None):
        super(NeuralInputs, self).__init__(numberOfInputs, Extension, NeuralInput, )

    #
    # XMLBehaviors
    #
    def set_NeuralInput(self, NeuralInput, *args):
        self.NeuralInput = NeuralInput

        self.numberOfInputs = len(NeuralInput)
    def set_NeuralInput_wrapper(self, NeuralInput, *args):
        result = self.set_NeuralInput(NeuralInput, *args)
        return result

    def add_NeuralInput(self, value, *args):
        self.NeuralInput.append(value)

        self.numberOfInputs = len(self.NeuralInput)
    def add_NeuralInput_wrapper(self, value, *args):
        result = self.add_NeuralInput(value, *args)
        return result

    def insert_NeuralInput_at(self, index, value, *args):
        self.NeuralInput.insert(index, value)

        self.numberOfInputs = len(self.NeuralInput)
    def insert_NeuralInput_at_wrapper(self, index, value, *args):
        result = self.insert_NeuralInput_at(index, value, *args)
        return result

supermod.NeuralInputs.subclass = NeuralInputs
# end class NeuralInputs


class NeuralLayer(supermod.NeuralLayer):
    def __init__(self, numberOfNeurons=None, activationFunction=None, threshold=None, width=None, altitude=None, normalizationMethod=None, Extension=None, Neuron=None):
        super(NeuralLayer, self).__init__(numberOfNeurons, activationFunction, threshold, width, altitude, normalizationMethod, Extension, Neuron, )

    #
    # XMLBehaviors
    #
    def set_Neuron(self, Neuron, *args):
        self.Neuron = Neuron

        self.numberOfNeurons = len(self.Neuron)
    def set_Neuron_wrapper(self, Neuron, *args):
        result = self.set_Neuron(Neuron, *args)
        return result

    def add_Neuron(self, value, *args):
        self.Neuron.append(value)

        self.numberOfNeurons = len(self.Neuron)
    def add_Neuron_wrapper(self, value, *args):
        result = self.add_Neuron(value, *args)
        return result

    def insert_Neuron_at(self, index, value, *args):
        self.Neuron.insert(index, value)

        self.numberOfNeurons = len(self.Neuron)
    def insert_Neuron_at_wrapper(self, index, value, *args):
        result = self.insert_Neuron_at(index, value, *args)
        return result

supermod.NeuralLayer.subclass = NeuralLayer
# end class NeuralLayer


class NeuralOutputs(supermod.NeuralOutputs):
    def __init__(self, numberOfOutputs=None, Extension=None, NeuralOutput=None):
        super(NeuralOutputs, self).__init__(numberOfOutputs, Extension, NeuralOutput, )

    #
    # XMLBehaviors
    #
    def set_NeuralOutput(self, NeuralOutput, *args):
        self.Neuron = Neuron

        self.numberOfNeurons = len(self.Neuron)
    def set_NeuralOutput_wrapper(self, NeuralOutput, *args):
        result = self.set_NeuralOutput(NeuralOutput, *args)
        return result

    def add_NeuralOutput(self, value, *args):
         self.NeuralOutput.append(value)

         self.numberOfOutputs = len(self.NeuralOutput)
    def add_NeuralOutput_wrapper(self, value, *args):
        result = self.add_NeuralOutput(value, *args)
        return result

    def insert_NeuralOutput_at(self, index, value, *args):
        self.NeuralOutput.insert(index, value)

        self.numberOfOutputs = len(self.NeuralOutput)
    def insert_NeuralOutput_at_wrapper(self, index, value, *args):
        result = self.insert_NeuralOutput_at(index, value, *args)
        return result

supermod.NeuralOutputs.subclass = NeuralOutputs
# end class NeuralOutputs


class NeuralInput(supermod.NeuralInput):
    def __init__(self, id=None, Extension=None, DerivedField=None):
        super(NeuralInput, self).__init__(id, Extension, DerivedField, )

    #
    # XMLBehaviors
    #
supermod.NeuralInput.subclass = NeuralInput
# end class NeuralInput


class Neuron(supermod.Neuron):
    def __init__(self, id=None, bias=None, width=None, altitude=None, Extension=None, Con=None):
        super(Neuron, self).__init__(id, bias, width, altitude, Extension, Con, )

    #
    # XMLBehaviors
    #
supermod.Neuron.subclass = Neuron
# end class Neuron


class Con(supermod.Con):
    def __init__(self, from_=None, weight=None, Extension=None):
        super(Con, self).__init__(from_, weight, Extension, )

    #
    # XMLBehaviors
    #
supermod.Con.subclass = Con
# end class Con


class NeuralOutput(supermod.NeuralOutput):
    def __init__(self, outputNeuron=None, Extension=None, DerivedField=None):
        super(NeuralOutput, self).__init__(outputNeuron, Extension, DerivedField, )

    #
    # XMLBehaviors
    #
supermod.NeuralOutput.subclass = NeuralOutput
# end class NeuralOutput


class DataDictionary(supermod.DataDictionary):
    def __init__(self, numberOfFields=None, Extension=None, DataField=None, Taxonomy=None):
        super(DataDictionary, self).__init__(numberOfFields, Extension, DataField, Taxonomy, )

    #
    # XMLBehaviors
    #
    def set_DataField(self, DataField, *args):
        self.DataField = DataField

        self.numberOfFields = len(self.DataField)
    def set_DataField_wrapper(self, DataField, *args):
        result = self.set_DataField(DataField, *args)
        return result

    def add_DataField(self, value, *args):
        self.DataField.append(value)

        self.numberOfFields = len(self.DataField)
    def add_DataField_wrapper(self, value, *args):
        result = self.add_DataField(value, *args)
        return result

    def insert_DataField_at(self, index, value, *args):
        self.DataField.insert(index, value)

        self.numberOfFields = len(self.DataField)
    def insert_DataField_at_wrapper(self, index, value, *args):
        result = self.insert_DataField_at(index, value, *args)
        return result

supermod.DataDictionary.subclass = DataDictionary
# end class DataDictionary


class DataField(supermod.DataField):
    def __init__(self, name=None, displayName=None, optype=None, dataType=None, taxonomy=None, isCyclic='0', Extension=None, Interval=None, Value=None):
        super(DataField, self).__init__(name, displayName, optype, dataType, taxonomy, isCyclic, Extension, Interval, Value, )

    #
    # XMLBehaviors
    #
supermod.DataField.subclass = DataField
# end class DataField


class Value(supermod.Value):
    def __init__(self, value=None, displayValue=None, property='valid', Extension=None):
        super(Value, self).__init__(value, displayValue, property, Extension, )

    #
    # XMLBehaviors
    #
supermod.Value.subclass = Value
# end class Value


class Interval(supermod.Interval):
    def __init__(self, closure=None, leftMargin=None, rightMargin=None, Extension=None):
        super(Interval, self).__init__(closure, leftMargin, rightMargin, Extension, )

    #
    # XMLBehaviors
    #
supermod.Interval.subclass = Interval
# end class Interval


class Header(supermod.Header):
    def __init__(self, copyright=None, description=None, modelVersion=None, Extension=None, Application=None, Annotation=None, Timestamp=None):
        super(Header, self).__init__(copyright, description, modelVersion, Extension, Application, Annotation, Timestamp, )

    #
    # XMLBehaviors
    #
    def exportAttributes(self, outfile, level, already_processed, namespace_='', name_='Header', *args):
        from datetime import datetime

        

        if self.copyright is not None and 'copyright' not in already_processed:

            if not self.copyright.endswith("Software AG"):

                self.copyright += ", exported to PMML by Nyoka (c) " + str(datetime.now().year) + " Software AG"

            already_processed.add('copyright')

            outfile.write(' copyright=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.copyright), input_name='copyright')), ))

        if self.description is not None and 'description' not in already_processed:

            already_processed.add('description')

            outfile.write(' description=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.description), input_name='description')), ))

        if self.modelVersion is not None and 'modelVersion' not in already_processed:

            already_processed.add('modelVersion')

            outfile.write(' modelVersion=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.modelVersion), input_name='modelVersion')), ))


    def exportAttributes_wrapper(self, outfile, level, already_processed, namespace_='', name_='Header', *args):
        result = self.exportAttributes(outfile, level, already_processed, namespace_='', name_='Header', *args)
        return result

supermod.Header.subclass = Header
# end class Header


class Application(supermod.Application):
    def __init__(self, name=None, version=None, Extension=None):
        super(Application, self).__init__(name, version, Extension, )

    #
    # XMLBehaviors
    #
supermod.Application.subclass = Application
# end class Application


class Annotation(supermod.Annotation):
    def __init__(self, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        super(Annotation, self).__init__(Extension, valueOf_, mixedclass_, content_, )

    #
    # XMLBehaviors
    #
supermod.Annotation.subclass = Annotation
# end class Annotation


class Timestamp(supermod.Timestamp):
    def __init__(self, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        super(Timestamp, self).__init__(Extension, valueOf_, mixedclass_, content_, )

    #
    # XMLBehaviors
    #
    def export(self, outfile, level, namespace_='', name_='Timestamp', namespacedef_='', pretty_print=True, *args):
        imported_ns_def_ = supermod.GenerateDSNamespaceDefs_.get('Timestamp')

        if imported_ns_def_ is not None:

            namespacedef_ = imported_ns_def_

        if pretty_print:

            eol_ = '\n'

        else:

            eol_ = ''

        if self.original_tagname_ is not None:

            name_ = self.original_tagname_

        supermod.showIndent(outfile, level, pretty_print)

        outfile.write('<%s%s%s' % (namespace_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))

        already_processed = set()

        self.exportAttributes(outfile, level, already_processed, namespace_, name_='Timestamp')

        if self.hasContent_():

            outfile.write('>%s' % ('', ))

            self.exportChildren(outfile, level + 1, namespace_='', name_='Timestamp', pretty_print=pretty_print)

            supermod.showIndent(outfile, 0, pretty_print)

            outfile.write('</%s%s>%s' % (namespace_, name_, eol_))

        else:

            outfile.write('/>%s' % (eol_, ))

        
    def export_wrapper(self, outfile, level, namespace_='', name_='Timestamp', namespacedef_='', pretty_print=True, *args):
        result = self.export(outfile, level, namespace_='', name_='Timestamp', namespacedef_='', pretty_print=True, *args)
        return result

supermod.Timestamp.subclass = Timestamp
# end class Timestamp


class NaiveBayesModel(supermod.NaiveBayesModel):
    def __init__(self, modelName=None, threshold=None, functionName=None, algorithmName=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, BayesInputs=None, BayesOutput=None, ModelVerification=None, Extension=None):
        super(NaiveBayesModel, self).__init__(modelName, threshold, functionName, algorithmName, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, BayesInputs, BayesOutput, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.NaiveBayesModel.subclass = NaiveBayesModel
# end class NaiveBayesModel


class BayesInputs(supermod.BayesInputs):
    def __init__(self, Extension=None, BayesInput=None):
        super(BayesInputs, self).__init__(Extension, BayesInput, )

    #
    # XMLBehaviors
    #
supermod.BayesInputs.subclass = BayesInputs
# end class BayesInputs


class BayesInput(supermod.BayesInput):
    def __init__(self, fieldName=None, Extension=None, TargetValueStats=None, DerivedField=None, PairCounts=None):
        super(BayesInput, self).__init__(fieldName, Extension, TargetValueStats, DerivedField, PairCounts, )

    #
    # XMLBehaviors
    #
supermod.BayesInput.subclass = BayesInput
# end class BayesInput


class BayesOutput(supermod.BayesOutput):
    def __init__(self, fieldName=None, Extension=None, TargetValueCounts=None):
        super(BayesOutput, self).__init__(fieldName, Extension, TargetValueCounts, )

    #
    # XMLBehaviors
    #
supermod.BayesOutput.subclass = BayesOutput
# end class BayesOutput


class TargetValueStats(supermod.TargetValueStats):
    def __init__(self, Extension=None, TargetValueStat=None):
        super(TargetValueStats, self).__init__(Extension, TargetValueStat, )

    #
    # XMLBehaviors
    #
supermod.TargetValueStats.subclass = TargetValueStats
# end class TargetValueStats


class TargetValueStat(supermod.TargetValueStat):
    def __init__(self, value=None, Extension=None, AnyDistribution=None, GaussianDistribution=None, PoissonDistribution=None, UniformDistribution=None):
        super(TargetValueStat, self).__init__(value, Extension, AnyDistribution, GaussianDistribution, PoissonDistribution, UniformDistribution, )

    #
    # XMLBehaviors
    #
supermod.TargetValueStat.subclass = TargetValueStat
# end class TargetValueStat


class PairCounts(supermod.PairCounts):
    def __init__(self, value=None, Extension=None, TargetValueCounts=None):
        super(PairCounts, self).__init__(value, Extension, TargetValueCounts, )

    #
    # XMLBehaviors
    #
supermod.PairCounts.subclass = PairCounts
# end class PairCounts


class TargetValueCounts(supermod.TargetValueCounts):
    def __init__(self, Extension=None, TargetValueCount=None):
        super(TargetValueCounts, self).__init__(Extension, TargetValueCount, )

    #
    # XMLBehaviors
    #
supermod.TargetValueCounts.subclass = TargetValueCounts
# end class TargetValueCounts


class TargetValueCount(supermod.TargetValueCount):
    def __init__(self, value=None, count=None, Extension=None):
        super(TargetValueCount, self).__init__(value, count, Extension, )

    #
    # XMLBehaviors
    #
supermod.TargetValueCount.subclass = TargetValueCount
# end class TargetValueCount


class BaselineModel(supermod.BaselineModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, TestDistributions=None, ModelVerification=None, Extension=None):
        super(BaselineModel, self).__init__(modelName, functionName, algorithmName, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, TestDistributions, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.BaselineModel.subclass = BaselineModel
# end class BaselineModel


class TestDistributions(supermod.TestDistributions):
    def __init__(self, field=None, testStatistic=None, resetValue='0.0', windowSize='0', weightField=None, normalizationScheme=None, Extension=None, Baseline=None, Alternate=None):
        super(TestDistributions, self).__init__(field, testStatistic, resetValue, windowSize, weightField, normalizationScheme, Extension, Baseline, Alternate, )

    #
    # XMLBehaviors
    #
supermod.TestDistributions.subclass = TestDistributions
# end class TestDistributions


class Baseline(supermod.Baseline):
    def __init__(self, AnyDistribution=None, GaussianDistribution=None, PoissonDistribution=None, UniformDistribution=None, CountTable=None, NormalizedCountTable=None, FieldRef=None):
        super(Baseline, self).__init__(AnyDistribution, GaussianDistribution, PoissonDistribution, UniformDistribution, CountTable, NormalizedCountTable, FieldRef, )

    #
    # XMLBehaviors
    #
supermod.Baseline.subclass = Baseline
# end class Baseline


class Alternate(supermod.Alternate):
    def __init__(self, AnyDistribution=None, GaussianDistribution=None, PoissonDistribution=None, UniformDistribution=None):
        super(Alternate, self).__init__(AnyDistribution, GaussianDistribution, PoissonDistribution, UniformDistribution, )

    #
    # XMLBehaviors
    #
supermod.Alternate.subclass = Alternate
# end class Alternate


class AnyDistribution(supermod.AnyDistribution):
    def __init__(self, mean=None, variance=None, Extension=None):
        super(AnyDistribution, self).__init__(mean, variance, Extension, )

    #
    # XMLBehaviors
    #
supermod.AnyDistribution.subclass = AnyDistribution
# end class AnyDistribution


class GaussianDistribution(supermod.GaussianDistribution):
    def __init__(self, mean=None, variance=None, Extension=None):
        super(GaussianDistribution, self).__init__(mean, variance, Extension, )

    #
    # XMLBehaviors
    #
supermod.GaussianDistribution.subclass = GaussianDistribution
# end class GaussianDistribution


class PoissonDistribution(supermod.PoissonDistribution):
    def __init__(self, mean=None, Extension=None):
        super(PoissonDistribution, self).__init__(mean, Extension, )

    #
    # XMLBehaviors
    #
supermod.PoissonDistribution.subclass = PoissonDistribution
# end class PoissonDistribution


class UniformDistribution(supermod.UniformDistribution):
    def __init__(self, lower=None, upper=None, Extension=None):
        super(UniformDistribution, self).__init__(lower, upper, Extension, )

    #
    # XMLBehaviors
    #
supermod.UniformDistribution.subclass = UniformDistribution
# end class UniformDistribution


class COUNT_TABLE_TYPE(supermod.COUNT_TABLE_TYPE):
    def __init__(self, sample=None, Extension=None, FieldValue=None, FieldValueCount=None):
        super(COUNT_TABLE_TYPE, self).__init__(sample, Extension, FieldValue, FieldValueCount, )

    #
    # XMLBehaviors
    #
supermod.COUNT_TABLE_TYPE.subclass = COUNT_TABLE_TYPE
# end class COUNT_TABLE_TYPE


class FieldValue(supermod.FieldValue):
    def __init__(self, field=None, value=None, Extension=None, FieldValue_member=None, FieldValueCount=None):
        super(FieldValue, self).__init__(field, value, Extension, FieldValue_member, FieldValueCount, )

    #
    # XMLBehaviors
    #
supermod.FieldValue.subclass = FieldValue
# end class FieldValue


class FieldValueCount(supermod.FieldValueCount):
    def __init__(self, field=None, value=None, count=None, Extension=None):
        super(FieldValueCount, self).__init__(field, value, count, Extension, )

    #
    # XMLBehaviors
    #
supermod.FieldValueCount.subclass = FieldValueCount
# end class FieldValueCount


class BayesianNetworkModel(supermod.BayesianNetworkModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, modelType='General', inferenceMethod='Other', isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, BayesianNetworkNodes=None, ModelVerification=None, Extension=None):
        super(BayesianNetworkModel, self).__init__(modelName, functionName, algorithmName, modelType, inferenceMethod, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, BayesianNetworkNodes, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.BayesianNetworkModel.subclass = BayesianNetworkModel
# end class BayesianNetworkModel


class BayesianNetworkNodes(supermod.BayesianNetworkNodes):
    def __init__(self, Extension=None, DiscreteNode=None, ContinuousNode=None):
        super(BayesianNetworkNodes, self).__init__(Extension, DiscreteNode, ContinuousNode, )

    #
    # XMLBehaviors
    #
supermod.BayesianNetworkNodes.subclass = BayesianNetworkNodes
# end class BayesianNetworkNodes


class DiscreteNode(supermod.DiscreteNode):
    def __init__(self, name=None, count=None, Extension=None, DerivedField=None, DiscreteConditionalProbability=None, ValueProbability=None):
        super(DiscreteNode, self).__init__(name, count, Extension, DerivedField, DiscreteConditionalProbability, ValueProbability, )

    #
    # XMLBehaviors
    #
supermod.DiscreteNode.subclass = DiscreteNode
# end class DiscreteNode


class ContinuousNode(supermod.ContinuousNode):
    def __init__(self, name=None, count=None, Extension=None, DerivedField=None, ContinuousConditionalProbability=None, ContinuousDistribution=None):
        super(ContinuousNode, self).__init__(name, count, Extension, DerivedField, ContinuousConditionalProbability, ContinuousDistribution, )

    #
    # XMLBehaviors
    #
supermod.ContinuousNode.subclass = ContinuousNode
# end class ContinuousNode


class DiscreteConditionalProbability(supermod.DiscreteConditionalProbability):
    def __init__(self, count=None, Extension=None, ParentValue=None, ValueProbability=None):
        super(DiscreteConditionalProbability, self).__init__(count, Extension, ParentValue, ValueProbability, )

    #
    # XMLBehaviors
    #
supermod.DiscreteConditionalProbability.subclass = DiscreteConditionalProbability
# end class DiscreteConditionalProbability


class ParentValue(supermod.ParentValue):
    def __init__(self, parent=None, value=None, Extension=None):
        super(ParentValue, self).__init__(parent, value, Extension, )

    #
    # XMLBehaviors
    #
supermod.ParentValue.subclass = ParentValue
# end class ParentValue


class ValueProbability(supermod.ValueProbability):
    def __init__(self, value=None, probability=None, Extension=None):
        super(ValueProbability, self).__init__(value, probability, Extension, )

    #
    # XMLBehaviors
    #
supermod.ValueProbability.subclass = ValueProbability
# end class ValueProbability


class ContinuousConditionalProbability(supermod.ContinuousConditionalProbability):
    def __init__(self, count=None, Extension=None, ParentValue=None, ContinuousDistribution=None):
        super(ContinuousConditionalProbability, self).__init__(count, Extension, ParentValue, ContinuousDistribution, )

    #
    # XMLBehaviors
    #
supermod.ContinuousConditionalProbability.subclass = ContinuousConditionalProbability
# end class ContinuousConditionalProbability


class ContinuousDistribution(supermod.ContinuousDistribution):
    def __init__(self, Extension=None, TriangularDistributionForBN=None, NormalDistributionForBN=None, LognormalDistributionForBN=None, UniformDistributionForBN=None):
        super(ContinuousDistribution, self).__init__(Extension, TriangularDistributionForBN, NormalDistributionForBN, LognormalDistributionForBN, UniformDistributionForBN, )

    #
    # XMLBehaviors
    #
supermod.ContinuousDistribution.subclass = ContinuousDistribution
# end class ContinuousDistribution


class TriangularDistributionForBN(supermod.TriangularDistributionForBN):
    def __init__(self, Extension=None, Mean=None, Lower=None, Upper=None):
        super(TriangularDistributionForBN, self).__init__(Extension, Mean, Lower, Upper, )

    #
    # XMLBehaviors
    #
supermod.TriangularDistributionForBN.subclass = TriangularDistributionForBN
# end class TriangularDistributionForBN


class NormalDistributionForBN(supermod.NormalDistributionForBN):
    def __init__(self, Extension=None, Mean=None, Variance=None):
        super(NormalDistributionForBN, self).__init__(Extension, Mean, Variance, )

    #
    # XMLBehaviors
    #
supermod.NormalDistributionForBN.subclass = NormalDistributionForBN
# end class NormalDistributionForBN


class LognormalDistributionForBN(supermod.LognormalDistributionForBN):
    def __init__(self, Extension=None, Mean=None, Variance=None):
        super(LognormalDistributionForBN, self).__init__(Extension, Mean, Variance, )

    #
    # XMLBehaviors
    #
supermod.LognormalDistributionForBN.subclass = LognormalDistributionForBN
# end class LognormalDistributionForBN


class UniformDistributionForBN(supermod.UniformDistributionForBN):
    def __init__(self, Extension=None, Lower=None, Upper=None):
        super(UniformDistributionForBN, self).__init__(Extension, Lower, Upper, )

    #
    # XMLBehaviors
    #
supermod.UniformDistributionForBN.subclass = UniformDistributionForBN
# end class UniformDistributionForBN


class Mean(supermod.Mean):
    def __init__(self, Extension=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None):
        super(Mean, self).__init__(Extension, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.Mean.subclass = Mean
# end class Mean


class Lower(supermod.Lower):
    def __init__(self, Extension=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None):
        super(Lower, self).__init__(Extension, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.Lower.subclass = Lower
# end class Lower


class Upper(supermod.Upper):
    def __init__(self, Extension=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None):
        super(Upper, self).__init__(Extension, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.Upper.subclass = Upper
# end class Upper


class Variance(supermod.Variance):
    def __init__(self, Extension=None, FieldRef=None, Apply=None, Constant=None, NormContinuous=None, NormDiscrete=None, Discretize=None, MapValues=None, TextIndex=None, Aggregate=None, Lag=None):
        super(Variance, self).__init__(Extension, FieldRef, Apply, Constant, NormContinuous, NormDiscrete, Discretize, MapValues, TextIndex, Aggregate, Lag, )

    #
    # XMLBehaviors
    #
supermod.Variance.subclass = Variance
# end class Variance


class Targets(supermod.Targets):
    def __init__(self, Extension=None, Target=None):
        super(Targets, self).__init__(Extension, Target, )

    #
    # XMLBehaviors
    #
supermod.Targets.subclass = Targets
# end class Targets


class Target(supermod.Target):
    def __init__(self, field=None, optype=None, castInteger=None, min=None, max=None, rescaleConstant=0, rescaleFactor=1, Extension=None, TargetValue=None):
        super(Target, self).__init__(field, optype, castInteger, min, max, rescaleConstant, rescaleFactor, Extension, TargetValue, )

    #
    # XMLBehaviors
    #
supermod.Target.subclass = Target
# end class Target


class TargetValue(supermod.TargetValue):
    def __init__(self, value=None, displayValue=None, priorProbability=None, defaultValue=None, Extension=None, Partition=None):
        super(TargetValue, self).__init__(value, displayValue, priorProbability, defaultValue, Extension, Partition, )

    #
    # XMLBehaviors
    #
supermod.TargetValue.subclass = TargetValue
# end class TargetValue


class TextModel(supermod.TextModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, numberOfTerms=None, numberOfDocuments=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, TextDictionary=None, TextCorpus=None, DocumentTermMatrix=None, TextModelNormalization=None, TextModelSimiliarity=None, ModelVerification=None, Extension=None):
        super(TextModel, self).__init__(modelName, functionName, algorithmName, numberOfTerms, numberOfDocuments, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, TextDictionary, TextCorpus, DocumentTermMatrix, TextModelNormalization, TextModelSimiliarity, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.TextModel.subclass = TextModel
# end class TextModel


class TextDictionary(supermod.TextDictionary):
    def __init__(self, Extension=None, Taxonomy=None, Array=None):
        super(TextDictionary, self).__init__(Extension, Taxonomy, Array, )

    #
    # XMLBehaviors
    #
supermod.TextDictionary.subclass = TextDictionary
# end class TextDictionary


class TextCorpus(supermod.TextCorpus):
    def __init__(self, Extension=None, TextDocument=None):
        super(TextCorpus, self).__init__(Extension, TextDocument, )

    #
    # XMLBehaviors
    #
supermod.TextCorpus.subclass = TextCorpus
# end class TextCorpus


class TextDocument(supermod.TextDocument):
    def __init__(self, id=None, name=None, length=None, file=None, Extension=None):
        super(TextDocument, self).__init__(id, name, length, file, Extension, )

    #
    # XMLBehaviors
    #
supermod.TextDocument.subclass = TextDocument
# end class TextDocument


class DocumentTermMatrix(supermod.DocumentTermMatrix):
    def __init__(self, Extension=None, Matrix=None):
        super(DocumentTermMatrix, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.DocumentTermMatrix.subclass = DocumentTermMatrix
# end class DocumentTermMatrix


class TextModelNormalization(supermod.TextModelNormalization):
    def __init__(self, localTermWeights='termFrequency', globalTermWeights='inverseDocumentFrequency', documentNormalization='none', Extension=None):
        super(TextModelNormalization, self).__init__(localTermWeights, globalTermWeights, documentNormalization, Extension, )

    #
    # XMLBehaviors
    #
supermod.TextModelNormalization.subclass = TextModelNormalization
# end class TextModelNormalization


class TextModelSimiliarity(supermod.TextModelSimiliarity):
    def __init__(self, similarityType=None, Extension=None):
        super(TextModelSimiliarity, self).__init__(similarityType, Extension, )

    #
    # XMLBehaviors
    #
supermod.TextModelSimiliarity.subclass = TextModelSimiliarity
# end class TextModelSimiliarity


class ClusteringModel(supermod.ClusteringModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, modelClass=None, numberOfClusters=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, LocalTransformations=None, ComparisonMeasure=None, ClusteringField=None, MissingValueWeights=None, Cluster=None, ModelVerification=None, Extension=None):
        super(ClusteringModel, self).__init__(modelName, functionName, algorithmName, modelClass, numberOfClusters, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, LocalTransformations, ComparisonMeasure, ClusteringField, MissingValueWeights, Cluster, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
    def set_Cluster(self, Cluster, *args):
        self.Cluster = Cluster

        self.numberOfClusters = len(self.Cluster)
    def set_Cluster_wrapper(self, Cluster, *args):
        result = self.set_Cluster(Cluster, *args)
        return result

    def add_Cluster(self, value, *args):
        self.Cluster.append(value)

        self.numberOfClusters = len(self.Cluster)
    def add_Cluster_wrapper(self, value, *args):
        result = self.add_Cluster(value, *args)
        return result

    def insert_Cluster_at(self, index, value, *args):
        self.Cluster.insert(index, value)

        self.numberOfClusters = len(self.Cluster)
    def insert_Cluster_at_wrapper(self, index, value, *args):
        result = self.insert_Cluster_at(index, value, *args)
        return result

supermod.ClusteringModel.subclass = ClusteringModel
# end class ClusteringModel


class MissingValueWeights(supermod.MissingValueWeights):
    def __init__(self, Extension=None, Array=None):
        super(MissingValueWeights, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.MissingValueWeights.subclass = MissingValueWeights
# end class MissingValueWeights


class Cluster(supermod.Cluster):
    def __init__(self, id=None, name=None, size=None, Extension=None, KohonenMap=None, Array=None, Partition=None, Covariances=None):
        super(Cluster, self).__init__(id, name, size, Extension, KohonenMap, Array, Partition, Covariances, )

    #
    # XMLBehaviors
    #
supermod.Cluster.subclass = Cluster
# end class Cluster


class KohonenMap(supermod.KohonenMap):
    def __init__(self, coord1=None, coord2=None, coord3=None, Extension=None):
        super(KohonenMap, self).__init__(coord1, coord2, coord3, Extension, )

    #
    # XMLBehaviors
    #
supermod.KohonenMap.subclass = KohonenMap
# end class KohonenMap


class Covariances(supermod.Covariances):
    def __init__(self, Extension=None, Matrix=None):
        super(Covariances, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.Covariances.subclass = Covariances
# end class Covariances


class ClusteringField(supermod.ClusteringField):
    def __init__(self, field=None, isCenterField='true', fieldWeight='1', similarityScale=None, compareFunction=None, Extension=None, Comparisons=None):
        super(ClusteringField, self).__init__(field, isCenterField, fieldWeight, similarityScale, compareFunction, Extension, Comparisons, )

    #
    # XMLBehaviors
    #
supermod.ClusteringField.subclass = ClusteringField
# end class ClusteringField


class Comparisons(supermod.Comparisons):
    def __init__(self, Extension=None, Matrix=None):
        super(Comparisons, self).__init__(Extension, Matrix, )

    #
    # XMLBehaviors
    #
supermod.Comparisons.subclass = Comparisons
# end class Comparisons


class ComparisonMeasure(supermod.ComparisonMeasure):
    def __init__(self, kind=None, compareFunction='absDiff', minimum=None, maximum=None, Extension=None, euclidean=None, squaredEuclidean=None, chebychev=None, cityBlock=None, minkowski=None, simpleMatching=None, jaccard=None, tanimoto=None, binarySimilarity=None):
        super(ComparisonMeasure, self).__init__(kind, compareFunction, minimum, maximum, Extension, euclidean, squaredEuclidean, chebychev, cityBlock, minkowski, simpleMatching, jaccard, tanimoto, binarySimilarity, )

    #
    # XMLBehaviors
    #
supermod.ComparisonMeasure.subclass = ComparisonMeasure
# end class ComparisonMeasure


class euclidean(supermod.euclidean):
    def __init__(self, Extension=None):
        super(euclidean, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.euclidean.subclass = euclidean
# end class euclidean


class squaredEuclidean(supermod.squaredEuclidean):
    def __init__(self, Extension=None):
        super(squaredEuclidean, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.squaredEuclidean.subclass = squaredEuclidean
# end class squaredEuclidean


class cityBlock(supermod.cityBlock):
    def __init__(self, Extension=None):
        super(cityBlock, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.cityBlock.subclass = cityBlock
# end class cityBlock


class chebychev(supermod.chebychev):
    def __init__(self, Extension=None):
        super(chebychev, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.chebychev.subclass = chebychev
# end class chebychev


class minkowski(supermod.minkowski):
    def __init__(self, p_parameter=None, Extension=None):
        super(minkowski, self).__init__(p_parameter, Extension, )

    #
    # XMLBehaviors
    #
supermod.minkowski.subclass = minkowski
# end class minkowski


class simpleMatching(supermod.simpleMatching):
    def __init__(self, Extension=None):
        super(simpleMatching, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.simpleMatching.subclass = simpleMatching
# end class simpleMatching


class jaccard(supermod.jaccard):
    def __init__(self, Extension=None):
        super(jaccard, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.jaccard.subclass = jaccard
# end class jaccard


class tanimoto(supermod.tanimoto):
    def __init__(self, Extension=None):
        super(tanimoto, self).__init__(Extension, )

    #
    # XMLBehaviors
    #
supermod.tanimoto.subclass = tanimoto
# end class tanimoto


class binarySimilarity(supermod.binarySimilarity):
    def __init__(self, c00_parameter=None, c01_parameter=None, c10_parameter=None, c11_parameter=None, d00_parameter=None, d01_parameter=None, d10_parameter=None, d11_parameter=None, Extension=None):
        super(binarySimilarity, self).__init__(c00_parameter, c01_parameter, c10_parameter, c11_parameter, d00_parameter, d01_parameter, d10_parameter, d11_parameter, Extension, )

    #
    # XMLBehaviors
    #
supermod.binarySimilarity.subclass = binarySimilarity
# end class binarySimilarity


class GaussianProcessModel(supermod.GaussianProcessModel):
    def __init__(self, modelName=None, functionName=None, algorithmName=None, optimizer=None, isScorable=True, MiningSchema=None, Output=None, ModelStats=None, ModelExplanation=None, Targets=None, LocalTransformations=None, RadialBasisKernel=None, ARDSquaredExponentialKernel=None, AbsoluteExponentialKernel=None, GeneralizedExponentialKernel=None, TrainingInstances=None, ModelVerification=None, Extension=None):
        super(GaussianProcessModel, self).__init__(modelName, functionName, algorithmName, optimizer, isScorable, MiningSchema, Output, ModelStats, ModelExplanation, Targets, LocalTransformations, RadialBasisKernel, ARDSquaredExponentialKernel, AbsoluteExponentialKernel, GeneralizedExponentialKernel, TrainingInstances, ModelVerification, Extension, )

    #
    # XMLBehaviors
    #
supermod.GaussianProcessModel.subclass = GaussianProcessModel
# end class GaussianProcessModel


class RadialBasisKernel(supermod.RadialBasisKernel):
    def __init__(self, description=None, gamma='1', noiseVariance='1', lambda_='1', Extension=None):
        super(RadialBasisKernel, self).__init__(description, gamma, noiseVariance, lambda_, Extension, )

    #
    # XMLBehaviors
    #
supermod.RadialBasisKernel.subclass = RadialBasisKernel
# end class RadialBasisKernel


class ARDSquaredExponentialKernel(supermod.ARDSquaredExponentialKernel):
    def __init__(self, description=None, gamma='1', noiseVariance='1', Extension=None, Lambda=None):
        super(ARDSquaredExponentialKernel, self).__init__(description, gamma, noiseVariance, Extension, Lambda, )

    #
    # XMLBehaviors
    #
supermod.ARDSquaredExponentialKernel.subclass = ARDSquaredExponentialKernel
# end class ARDSquaredExponentialKernel


class AbsoluteExponentialKernel(supermod.AbsoluteExponentialKernel):
    def __init__(self, description=None, gamma='1', noiseVariance='1', Extension=None, Lambda=None):
        super(AbsoluteExponentialKernel, self).__init__(description, gamma, noiseVariance, Extension, Lambda, )

    #
    # XMLBehaviors
    #
supermod.AbsoluteExponentialKernel.subclass = AbsoluteExponentialKernel
# end class AbsoluteExponentialKernel


class GeneralizedExponentialKernel(supermod.GeneralizedExponentialKernel):
    def __init__(self, description=None, gamma='1', noiseVariance='1', degree='1', Extension=None, Lambda=None):
        super(GeneralizedExponentialKernel, self).__init__(description, gamma, noiseVariance, degree, Extension, Lambda, )

    #
    # XMLBehaviors
    #
supermod.GeneralizedExponentialKernel.subclass = GeneralizedExponentialKernel
# end class GeneralizedExponentialKernel


class Lambda(supermod.Lambda):
    def __init__(self, Extension=None, Array=None):
        super(Lambda, self).__init__(Extension, Array, )

    #
    # XMLBehaviors
    #
supermod.Lambda.subclass = Lambda
# end class Lambda


def get_root_tag(node):
    tag = supermod.Tag_pattern_.match(node.tag).groups()[-1]
    rootClass = None
    rootClass = supermod.GDSClassesMapping.get(tag)
    if rootClass is None and hasattr(supermod, tag):
        rootClass = getattr(supermod, tag)
    return tag, rootClass


def parseSub(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DefineFunction'
        rootClass = supermod.DefineFunction
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_='',
            pretty_print=True)
    return rootObj


def parseEtree(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DefineFunction'
        rootClass = supermod.DefineFunction
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
    mapping = {}
    rootElement = rootObj.to_etree(None, name_=rootTag, mapping_=mapping)
    reverse_mapping = rootObj.gds_reverse_node_mapping(mapping)
    if not silence:
        content = etree_.tostring(
            rootElement, pretty_print=True,
            xml_declaration=True, encoding="utf-8")
        sys.stdout.write(content)
        sys.stdout.write('\n')
    return rootObj, rootElement, mapping, reverse_mapping


def parseString(inString, silence=False):
    from StringIO import StringIO
    parser = None
    doc = parsexml_(StringIO(inString), parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DefineFunction'
        rootClass = supermod.DefineFunction
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_='')
    return rootObj


def parseLiteral(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DefineFunction'
        rootClass = supermod.DefineFunction
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
    if not silence:
        sys.stdout.write('#from nyoka.PMML44Super import *\n\n')
        sys.stdout.write('import nyoka.PMML44Super as model_\n\n')
        sys.stdout.write('rootObj = model_.rootClass(\n')
        rootObj.exportLiteral(sys.stdout, 0, name_=rootTag)
        sys.stdout.write(')\n')
    return rootObj


USAGE_TEXT = """
Usage: python ???.py <infilename>
"""


def usage():
    print(USAGE_TEXT)
    sys.exit(1)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        usage()
    infilename = args[0]
    parse(infilename)


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main()
def parse(inFileName, silence=False):
    orig_init()
    result = parseSub(inFileName, silence)
    new_init()
    
    return result

def new_init():

    def ArrayType_init(self, content=None, n=None, type_=None, mixedclass_=None):
        self.original_tagname_ = None
        self.n = supermod._cast(None, n)
        self.type_ = supermod._cast(None, type_)
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        self.content_ = [supermod.MixedContainer(1, 2, "", str(content))]
        self.valueOf_ = str(content)

    def Annotation_init(self, content=None, Extension=None, mixedclass_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        self.content_ = [supermod.MixedContainer(1, 2, "", str(content))]
        self.valueOf_ = str(content)
    
    def Timestamp_init(self, content=None, Extension=None, mixedclass_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        self.content_ = [supermod.MixedContainer(1, 2, "", str(content))]
        self.valueOf_ = str(content)

    def PMML_init(self, version='4.4', Header=None, MiningBuildTask=None, DataDictionary=None, TransformationDictionary=None, AssociationModel=None, AnomalyDetectionModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, Extension=None):
        self.original_tagname_ = None
        self.version = supermod._cast(None, version)
        self.Header = Header
        self.MiningBuildTask = MiningBuildTask
        self.DataDictionary = DataDictionary
        self.TransformationDictionary = TransformationDictionary
        if AssociationModel is None:
            self.AssociationModel = []
        else:
            self.AssociationModel = AssociationModel
        if AnomalyDetectionModel is None:
                self.AnomalyDetectionModel = []
        else:
            self.AnomalyDetectionModel = AnomalyDetectionModel
        if BayesianNetworkModel is None:
            self.BayesianNetworkModel = []
        else:
            self.BayesianNetworkModel = BayesianNetworkModel
        if BaselineModel is None:
            self.BaselineModel = []
        else:
            self.BaselineModel = BaselineModel
        if ClusteringModel is None:
            self.ClusteringModel = []
        else:
            self.ClusteringModel = ClusteringModel
        if GaussianProcessModel is None:
            self.GaussianProcessModel = []
        else:
            self.GaussianProcessModel = GaussianProcessModel
        if GeneralRegressionModel is None:
            self.GeneralRegressionModel = []
        else:
            self.GeneralRegressionModel = GeneralRegressionModel
        if MiningModel is None:
            self.MiningModel = []
        else:
            self.MiningModel = MiningModel
        if NaiveBayesModel is None:
            self.NaiveBayesModel = []
        else:
            self.NaiveBayesModel = NaiveBayesModel
        if NearestNeighborModel is None:
            self.NearestNeighborModel = []
        else:
            self.NearestNeighborModel = NearestNeighborModel
        if NeuralNetwork is None:
            self.NeuralNetwork = []
        else:
            self.NeuralNetwork = NeuralNetwork
        if RegressionModel is None:
            self.RegressionModel = []
        else:
            self.RegressionModel = RegressionModel
        if RuleSetModel is None:
            self.RuleSetModel = []
        else:
            self.RuleSetModel = RuleSetModel
        if SequenceModel is None:
            self.SequenceModel = []
        else:
            self.SequenceModel = SequenceModel
        if Scorecard is None:
            self.Scorecard = []
        else:
            self.Scorecard = Scorecard
        if SupportVectorMachineModel is None:
            self.SupportVectorMachineModel = []
        else:
            self.SupportVectorMachineModel = SupportVectorMachineModel
        if TextModel is None:
            self.TextModel = []
        else:
            self.TextModel = TextModel
        if TimeSeriesModel is None:
            self.TimeSeriesModel = []
        else:
            self.TimeSeriesModel = TimeSeriesModel
        if TreeModel is None:
            self.TreeModel = []
        else:
            self.TreeModel = TreeModel
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension

    ArrayType.__init__ = ArrayType_init
    Annotation.__init__ = Annotation_init
    Timestamp.__init__ = Timestamp_init
    PMML.__init__ = PMML_init

def orig_init():

    def ArrayType_init(self, n=None, type_=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        self.n = supermod._cast(None, n)
        self.type_ = supermod._cast(None, type_)
        self.valueOf_ = valueOf_
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    def Annotation_init(self, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        self.valueOf_ = valueOf_
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    def Timestamp_init(self, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        self.valueOf_ = valueOf_
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    def PMML_init(self, version=None, Header=None, MiningBuildTask=None, DataDictionary=None, TransformationDictionary=None, AssociationModel=None, AnomalyDetectionModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, Extension=None):
        self.original_tagname_ = None
        self.version = supermod._cast(None, version)
        self.Header = Header
        self.MiningBuildTask = MiningBuildTask
        self.DataDictionary = DataDictionary
        self.TransformationDictionary = TransformationDictionary
        if AssociationModel is None:
            self.AssociationModel = []
        else:
            self.AssociationModel = AssociationModel
        if AnomalyDetectionModel is None:
                self.AnomalyDetectionModel = []
        else:
            self.AnomalyDetectionModel = AnomalyDetectionModel
        if BayesianNetworkModel is None:
            self.BayesianNetworkModel = []
        else:
            self.BayesianNetworkModel = BayesianNetworkModel
        if BaselineModel is None:
            self.BaselineModel = []
        else:
            self.BaselineModel = BaselineModel
        if ClusteringModel is None:
            self.ClusteringModel = []
        else:
            self.ClusteringModel = ClusteringModel
        if GaussianProcessModel is None:
            self.GaussianProcessModel = []
        else:
            self.GaussianProcessModel = GaussianProcessModel
        if GeneralRegressionModel is None:
            self.GeneralRegressionModel = []
        else:
            self.GeneralRegressionModel = GeneralRegressionModel
        if MiningModel is None:
            self.MiningModel = []
        else:
            self.MiningModel = MiningModel
        if NaiveBayesModel is None:
            self.NaiveBayesModel = []
        else:
            self.NaiveBayesModel = NaiveBayesModel
        if NearestNeighborModel is None:
            self.NearestNeighborModel = []
        else:
            self.NearestNeighborModel = NearestNeighborModel
        if NeuralNetwork is None:
            self.NeuralNetwork = []
        else:
            self.NeuralNetwork = NeuralNetwork
        if RegressionModel is None:
            self.RegressionModel = []
        else:
            self.RegressionModel = RegressionModel
        if RuleSetModel is None:
            self.RuleSetModel = []
        else:
            self.RuleSetModel = RuleSetModel
        if SequenceModel is None:
            self.SequenceModel = []
        else:
            self.SequenceModel = SequenceModel
        if Scorecard is None:
            self.Scorecard = []
        else:
            self.Scorecard = Scorecard
        if SupportVectorMachineModel is None:
            self.SupportVectorMachineModel = []
        else:
            self.SupportVectorMachineModel = SupportVectorMachineModel
        if TextModel is None:
            self.TextModel = []
        else:
            self.TextModel = TextModel
        if TimeSeriesModel is None:
            self.TimeSeriesModel = []
        else:
            self.TimeSeriesModel = TimeSeriesModel
        if TreeModel is None:
            self.TreeModel = []
        else:
            self.TreeModel = TreeModel
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension

    ArrayType.__init__ = ArrayType_init
    Annotation.__init__ = Annotation_init
    Timestamp.__init__ = Timestamp_init
    PMML.__init__ = PMML_init

new_init()

def showIndent(outfile, level, pretty_print=True):
    if pretty_print:
        for idx in range(level):
            outfile.write('\t')
