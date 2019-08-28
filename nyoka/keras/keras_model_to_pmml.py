#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes used in keras_model_to_pmml.py

"""

from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

# python imports

import datetime
import json
import numpy as np

# nyoka imports
import PMML44 as ny
import metadata
import warnings
import base64
warnings.formatwarning = warnings.formatwarning = lambda msg, *args, **kwargs: str(msg)+'\n'

KERAS_LAYER_TYPES_MAP = {'InputLayer': 'Input',
                         'Add': 'MergeLayer',
                         'Concatenate': 'MergeLayer',
                         'Dot': 'MergeLayer',
                         'Subtract': 'MergeLayer',
                         'Maximum': 'MergeLayer',
                         'Minimum': 'MergeLayer',
                         'Average': 'MergeLayer'}

KERAS_LAYER_PARAMS = ['filters', 'kernel_size', 'strides', 'padding',
                      'input_shape', 'output_shape', "activation", "axis",
                      "epsilon", "pool_size", "scale", "center", "depth_multiplier",
                      "rate", "dilation_rate","size","stride","ratios","scales","mean","std","nms_threshold",
                      "score_threshold"]

NYOKA_LAYER_PARAMS = ['featureMaps', 'kernel', 'stride', 'pad',
                      'inputDimension', 'outputDimension',
                      "activationFunction", "axis",
                      "batchNormalizationEpsilon", "poolSize",
                      "batchNormalizationScale", "batchNormalizationCenter", "depthMultiplier",
                      "dropoutRate", "dilationRate","upsamplingSize","anchorStride","anchorRatios","anchorScales",
                      "regressBoxesMean", "regressBoxesStd","nmsThreshold","scoreThreshold"]


class KerasHeader(ny.Header):
    """
    Creates header for Keras PMML model file using Nyoka

    Parameters
    ----------
    copyright : String
        Adds the information about the copyright.
    description : String
        Description of the PMML file provided as a default

    Returns
    -------
    Nyoka header object
    """ 
 
    def __init__(self, description, copyright):
        if not description:
            description = "Keras Model in PMML"
        if not copyright:
            copyright = "Copyright (c) 2018 Software AG"
        ny.Header.__init__(self, copyright=copyright,
                           description=description,
                           Timestamp=ny.Timestamp(str(datetime.datetime.now())),
                           Application=ny.Application(name="Nyoka",version=metadata.__version__))


class KerasNetworkLayer(ny.NetworkLayer):
    """
    Creates Networklayer of PMML which has information about the layer type, weight matrix and bias matrix and their properties.

    Parameters
    ----------
    layer : Keras layer object
        Keras layer object
    dataSet : String
        Name of the dataset
    layer_type : String
        Class name of the layer
    connection_layer_id : boolean
        Whether to generate connection layer IDs or not

    Returns
    -------
    Nyoka NetworkLayer object
    """ 

    def _get_flatten_weights(self, weights):

        """
        Flattens the input

        Parameters
        ----------
        weights : array
            Array of weights

        Returns
        -------
        flat_weights : array
            Flattened input array in Base64String format

        weights_shape : array
            Shape of the flattened array
        """ 
        flat_weights = []
        weights_shape = []
        if weights and isinstance(weights, list):
            for item in weights:
                weights_shape.append(str(item.shape))
                flat_weights.append(item.flatten())
            flat_weights = np.concatenate(flat_weights).ravel()
            if len(weights) > 1:
                weights_shape = str((len(weights_shape), item.shape[0], 1))
            else:
                weights_shape = ", ".join(weights_shape)
        return flat_weights, weights_shape

    def _get_enumerated_input_shape(self, input_shape):
        """
        Gets the input shape from the Keras Input layer 

        Parameters
        ----------
        input_shape : array
            Array of shape

        Returns
        -------
        input_dims : array
            Array of shape for Nyoka PMML
        """ 
        input_dims = None
        if isinstance(input_shape, tuple):
            in_s = input_shape[1:]
            if len(in_s) == 1:
                input_dims = str((in_s[0], 1))
            else:
                input_dims = str(in_s)
        elif isinstance(input_shape, list):
            new_shape_lst = []
            for i_shape in input_shape:
                new_shape_lst.append(str(tuple(i_shape[1:])))
            input_dims = ", ".join(new_shape_lst)
        return input_dims

    def _get_activation_function(self, layer):
        """
        Identifies the Activation Function from a given Keras layer

        Parameters
        ----------
        layer : Keras layer object
            A Keras Layer

        Returns
        -------
        activation_function : String
            Activation function of the given Keras layer

        """ 
        layer_config = layer.get_config()
        if 'activation' in layer_config:
            activation_function = layer_config.get('activation')
            # if activation_function == "sigmoid":
            #     activation_function = "logistic"
            if activation_function == "relu":
                activation_function = "rectifier"
            elif activation_function == "relu6":
                activation_function = "reLU6"
            elif activation_function == "tanh":
                activation_function = "tanch"

        else:
            activation_function = None
        return activation_function

    def _get_layer_params_dict(self, layer):
        """
        Pull out the relevant Nyoka layer attributes matching with Keras layer attributes and pulls values for the respective attributes

        Parameters
        ----------
        layer : Keras layer object
            A Keras Layer

        Returns
        -------
        new_layer_params_dict : Dictionary
            Nyoka Layer attributes in a dictionary format

        """ 
        layer_params_dict = dict(zip(NYOKA_LAYER_PARAMS, KERAS_LAYER_PARAMS))
        layer_config = layer.get_config()
        new_layer_params_dict = {}
        pad_dims = None
        layer_params_dict['paddingDims']='None'  
        for key, val in layer_params_dict.items():
            if val in layer_config:
                if val == "activation":
                    layer_params_dict[key] = self._get_activation_function(layer)
                elif val == "padding":
                    pad_val = layer_config.get(val)
                    if pad_val in ['valid','same']:
                        layer_params_dict[key] = str(pad_val)
                    else:
                        pad_dims=str(pad_val)
                        layer_params_dict['paddingDims'] = pad_dims
                else:
                    layer_params_dict[key] = str(tuple(layer_config.get(val)))\
                         if layer_config.get(val).__class__.__name__ == 'list' else str(layer_config.get(val))
            elif hasattr(layer, val):
                if val == "input_shape":
                    try:
                        shape = getattr(layer, val)
                    except:
                        shape = layer.get_input_shape_at(0) 
                    layer_params_dict[key] = self._get_enumerated_input_shape(shape)
                elif val == "output_shape":
                    try:
                        shape = getattr(layer, val)
                    except:
                        shape = layer.get_output_shape_at(0) 
                    layer_params_dict[key] = self._get_enumerated_input_shape(shape)
                else:
                    layer_params_dict[key] = str(tuple(getattr(layer,val)))\
                         if getattr(layer,val).__class__.__name__ == 'list' else str(getattr(layer,val))
            else:
                layer_params_dict[key] = None
            if layer_params_dict[key] and layer_params_dict[key] != "None":
                if key == "pad" and pad_dims:
                    pad_ = list()
                    for val in pad_val:
                        if hasattr(val,'__len__'):
                            for v in val:
                                pad_.append(v)
                        else:
                            pad_.append(val)
                    pad_val = tuple(pad_)
                    new_layer_params_dict["paddingDims"] = str(pad_val)
                else:
                    new_layer_params_dict[key] = layer_params_dict[key]
        return new_layer_params_dict

    def _get_layer_weights_n_biases(self, layer):
        """
        Pulls out the Weights and Bias matrix from a given Keras layer

        Parameters
        ----------
        layer : Keras layer object
            A Keras Layer

        Returns
        -------
        layer_weights : array
            Weights of the Keras layer in Base64String format
        layer_biases : array
            Bias of the Keras layer in Base64String format

        """ 
        layer_all_weights = layer.get_weights()
        layer_weights = layer_biases = biases = None
        if layer_all_weights:
            if hasattr(layer, 'use_bias') and layer.use_bias:
                biases = layer_all_weights[-1]
                weights, w_shape = self._get_flatten_weights(
                    layer_all_weights[0:-1])
                layer_weights = ny.LayerWeights(content=weights,
                                                floatsPerLine=0,
                                                weightsShape=w_shape,
                                                weightsFlattenAxis="0")
            else:
                weights, w_shape = self._get_flatten_weights(layer_all_weights)
                layer_weights = ny.LayerWeights(content=weights,
                                                floatsPerLine=0,
                                                weightsShape=w_shape,
                                                weightsFlattenAxis="0")

            if biases is not None:
                bs_shape = biases.shape
                if len(bs_shape) == 1:
                    final_bs_shape = str((bs_shape[0], 1))
                else:
                    final_bs_shape = str(bs_shape)
                layer_biases = ny.LayerBias(content=biases,
                                            biasShape=final_bs_shape,
                                            biasFlattenAxis="0",
                                            floatsPerLine=0)
        return layer_weights, layer_biases

    def _get_connection_layer_ids(self, layer):
        """
        Pulls out the Connection ID of the Keras layer

        Parameters
        ----------
        layer : Keras layer object
            A Keras Layer

        Returns
        -------
        connection_layers : String
            The Layer ID of the Keras Layer

        """ 
        node_config = layer._inbound_nodes[0].get_config()
        if node_config['inbound_layers']:
            inbound_layers = node_config['inbound_layers']
            connection_layers = ", ".join(inbound_layers)
        else:
            connection_layers = "na"
        return connection_layers

    def __init__(self, layer,dataSet, layer_type, script_args, connection_layer_id=True):
        merge_layer_op_type = None
        merge_concat_axes = None
        merge_dot_axes = None
        merge_dot_normalization = False
        connection_layers = ''
        input_field_name = None
        if "Pmml" in layer_type:
            layer_type = layer_type[4:]
        old_layer_type = layer_type
        layer_type = KERAS_LAYER_TYPES_MAP.get(layer_type, layer_type)
        layer_params = self._get_layer_params_dict(layer)
        layer_weights, layer_biases = self._get_layer_weights_n_biases(layer)
        if connection_layer_id:
            connection_layers = self._get_connection_layer_ids(layer)
        if layer_type == "MergeLayer":
            merge_layer_op_type = old_layer_type.lower()
            if merge_layer_op_type == "concatenate":
                if "axis" in layer_params:
                    merge_concat_axes = layer_params["axis"]
                    del layer_params["axis"]
            elif merge_layer_op_type == 'dot':
                if 'axes' in layer_params:
                    merge_dot_axes = layer_params["axes"]
                    del layer_params["axes"]
                if 'normalize' in layer_params:
                    merge_dot_normalization = layer_params["normalize"]
                    del layer_params["normalize"]
                layer_params["mergeLayerDotNormalize"] = merge_dot_normalization
        elif layer_type == "BatchNormalization":
            layer_type = "BatchNormalization"
            if "axis" in layer_params:
                layer_params["batchNormalizationAxis"] = layer_params["axis"]
                del layer_params["axis"]
            if "batchNormalizationScale" in layer_params:
                try:
                    layer_params["batchNormalizationScale"] = eval(layer_params[
                        "batchNormalizationScale"])
                except:
                    layer_params["batchNormalizationScale"] = layer_params[
                        "batchNormalizationScale"]
            if "batchNormalizationCenter" in layer_params:
                try:
                    layer_params["batchNormalizationCenter"] = eval(layer_params[
                        "batchNormalizationCenter"])
                except:
                    layer_params["batchNormalizationCenter"] = layer_params[
                        "batchNormalizationCenter"]
        elif layer_type == "ReLU":
            layer_type = "Activation"
            layer_params["activationFunction"] = "reLU6"
        elif layer_type == "Anchors":
            layer_params["anchorSize"] = layer_params["upsamplingSize"]
            del layer_params["upsamplingSize"]
            layer_params["anchorRatios"] = layer_params["anchorRatios"]
            layer_params["anchorScales"] = layer_params["anchorScales"]
        elif layer_type == "RegressBoxes":
            layer_params["regressBoxesMean"] = layer_params["regressBoxesMean"]
            layer_params["regressBoxesStd"] = layer_params["regressBoxesStd"]
        elif layer_type == 'FilterDetections':
            layer_params['nms'] = layer.nms
            layer_params['classSpecificFilter'] = layer.class_specific_filter
        layer_params["mergeLayerOp"] = merge_layer_op_type
        layer_params["mergeLayerConcatOperationAxes"] = merge_concat_axes
        layer_params["mergeLayerDotOperationAxis"] = merge_dot_axes
        if layer_type == "Input":
            if dataSet=='image' or script_args:
                input_field_name = 'base64String'
            else:
                input_field_name = dataSet
        ny.NetworkLayer.__init__(self, inputFieldName=input_field_name,
                                 layerType=layer_type,
                                 connectionLayerId=connection_layers,
                                 layerId=layer.name,
                                 normalizationMethod="none",
                                 LayerParameters=ny.LayerParameters(
                                     **layer_params),
                                 LayerWeights=layer_weights,
                                 LayerBias=layer_biases)


class KerasDataDictionary(ny.DataDictionary):
    """
    KerasDataDictionary stores the class information to be predicted  in the PMML model.
    The current implementation takes care of the image class label by giving dataset name as dataSet parameter.

    Parameters
    ----------
    dataSet : String
        Name of the dataset
    predictedClasses : List
        List of class names or values to be predicted.
    Returns
    -------
    Nyoka's Dictionary Object
    """ 
    def __init__(self, dataSet, predictedClasses, script_args):
        ny.DataDictionary.__init__(self)
        if predictedClasses:
            class_node = ny.DataField(name="labels", optype="categorical",
                                    dataType="string")
            if type(predictedClasses) == list:
                if not all(type(pC) == str for pC in predictedClasses):
                    warnings.warn(
                        "Not all classes are given as String. Values will be attempted to be converted to String.")
                for i in range(len(predictedClasses)):
                    data_val = predictedClasses[i]
                    class_node.add_Value(ny.Value(value=str(data_val)))
            elif type(predictedClasses) == dict:
                if not all(type(pC) == str for pC in predictedClasses.keys()):
                    warnings.warn(
                        "Class indices are expected as strings in dictionary keys. Keys will be attempted to be converted to String.")
                for i in range(len(predictedClasses.keys())):
                    data_val = predictedClasses.keys()[i]
                    class_node.add_Value(ny.Value(value=str(data_val)))
        else:
            class_node = ny.DataField(name="predictions", optype="continuous",
                                        dataType="double")
        if dataSet == "image" or script_args:
            name = dataSet
            ny.DataDictionary.add_DataField(self, ny.DataField(
                name=name, optype="categorical", dataType="binary",
                mimeType="image/png", Extension=[ny.Extension(
                    extender="ADAPA", name="BINARY_BUFFERED", value="true")]))
        else:
            name = dataSet
            ny.DataDictionary.add_DataField(self, ny.DataField(
                name=name, optype="categorical", dataType="string"
            ))           

        ny.DataDictionary.add_DataField(self, class_node)


class KerasMiningSchema(ny.MiningSchema):
    """
    KerasMiningSchema stores the attributes which are used to build the model.
    
    Parameters
    ----------
    dataSet : String
        Name of the dataset

    Returns
    -------
    Nyoka's Mining Schema Object
    """ 
    def __init__(self, dataSet, predictedClasses):
        ny.MiningSchema.__init__(self)
        name = dataSet
        ny.MiningSchema.add_MiningField(self, ny.MiningField(
            name=name, usageType="active",
            invalidValueTreatment="asIs"))

        ny.MiningSchema.add_MiningField(self, ny.MiningField(
            name="labels" if predictedClasses else "predictions", usageType="target",
            invalidValueTreatment="asIs"))


class KerasOutput(ny.Output):
    """
    KerasOutput provides the information about the output representation of the PMML. (e.g. Predicted classes, probabilities)
    
    Parameters
    ----------
    predictedClasses : List
        List of Classes for which model has been trained

    Returns
    -------
    Nyoka's Output Object
    """ 
    def __init__(self, predictedClasses=None):
        ny.Output.__init__(self)
        if predictedClasses:
            ny.Output.add_OutputField(self, ny.OutputField(
                name="predicted_label", feature="predictedValue",
                dataType="string", optype="categorical"))
            ny.Output.add_OutputField(self, ny.OutputField(
                name="top1_prob", feature="probability", dataType="double"))
            ny.Output.add_OutputField(self, ny.OutputField(
                name="top5_prob", feature="topCategories", numTopCategories="5",
                dataType="string", optype="categorical"))
        else:
            ny.Output.add_OutputField(self, ny.OutputField(
                name="predicted_predictions", feature="predictedValue",
                dataType="double", optype="continuous"))



class KerasLocalTransformations(ny.LocalTransformations):
    """
    KerasLocalTransformations provides the information about the list of transformations applied to the data.
    
    Parameters
    ----------
    keras_model : Keras model object
        Keras model object
    dataSet : String
        Name of the dataset

    Returns
    -------
    Nyoka's Transformations Object
    """ 
    def __init__(self, keras_model, dataSet, script_args):
        ny.LocalTransformations.__init__(self)
        if script_args:
            ret_type = script_args['return_type'].lower()
            def_name = script_args['def_name'] if script_args['content'].__class__.__name__=='str' else script_args['content'].__name__

            ny.LocalTransformations.add_DerivedField(self, ny.DerivedField(
                name="base64String", optype="categorical" if ret_type=="string" else "continuous",
                dataType=ret_type, Apply=ny.Apply(function='customFunc',
                FieldRef=[ny.FieldRef(field=dataSet)])
            ))
        else:
            ny.LocalTransformations.add_DerivedField(self, ny.DerivedField(
                name="base64String", optype="categorical", dataType="string",
                Apply=ny.Apply(function="CNN:getBase64String",
                            FieldRef=[ny.FieldRef(field="image")])))


class KerasTransformationDictionary(ny.TransformationDictionary):
    """
    KerasTransformationDictionary provides the information about the list of transformations functions applied to the data.
    
    Parameters
    ----------
    dataSet : string
        name of the input
    script_args : Dictionary
        Arguments for the script

    Returns
    -------
    Nyoka's TransformationDictionary object 
    """
    def __init__(self, dataSet, script_args):
        if script_args['content'].__class__.__name__ == 'str':
            content = script_args['content']
            def_name = script_args['def_name']
        else:
            import inspect
            content = inspect.getsource(script_args['content'])
            def_name = script_args['content'].__name__
        encode = True
        if "encode" in script_args:
            encode = script_args['encode']
        
        if encode:
            content = base64.b64encode(content.encode()).decode()
        return_type = script_args['return_type'].lower()
        extension = [ny.Extension(
            extender='ADAPA', 
            name=def_name, 
            value=return_type,
            anytypeobjs_=[content]
            )]
        def_func = ny.DefineFunction(
            name='customFunc',
            optype='categorical' if return_type == 'string' else 'continous',
            dataType=return_type,
            ParameterField=[ny.ParameterField(name=dataSet, dataType='binary')],
            Apply = ny.Apply(function='python', Extension=extension, FieldRef=[ny.FieldRef(field=dataSet)]),

        )
        ny.TransformationDictionary.__init__(self)
        ny.TransformationDictionary.add_DefineFunction(self, def_func)




class KerasNetwork(ny.DeepNetwork):
    """
    KerasNetwork creates the DeepNetwork object which stores the NetworkLayer in sequence to define the architecture.
    
    Parameters
    ----------
    keras_model : Keras model object
        Keras model object
    model_name : String
        Name of the model
    dataSet : String
        Name of the dataset
    predictedClasses : List
        List of class names

    Returns
    -------
    Nyoka's DeepNetwork Object
    """ 


    def _create_an_input_layer(self, layer, dataSet, script_args):
        """
        Creates a PMML input layer from Keras Input Layer object
        
        Parameters
        ----------
        layer : Keras layer object
            Keras layer object
        dataSet : String
            Name of the dataset
    
        Returns
        -------
        input_layer: Nyoka Object
            PMML Input layer object
        """

        
        if dataSet=='image' or script_args:
            inputField = "base64String"
        else:
            inputField = dataSet

        in_shape = layer.input_shape
        if in_shape[0] is not None:
            if len(in_shape) == 1:
                input_dims = output_dims = str((in_shape[0],1))
            else:
                input_dims = output_dims = str(in_shape)  
        else:
            if len(in_shape) == 2:
                input_dims = output_dims = str(tuple(list(in_shape[1:])+[1]))
            else:
                input_dims = output_dims = str(tuple(list(in_shape[1:])))  
        node_config = layer._inbound_nodes[0].get_config()
        connection_layers = ", ".join(node_config['inbound_layers'])
        input_layer = ny.NetworkLayer(
            inputFieldName=inputField, layerType="Input", layerId=connection_layers,
            connectionLayerId="na",LayerParameters=ny.LayerParameters(
                inputDimension=input_dims,
                outputDimension=output_dims))
        return input_layer

    def _create_layers(self, keras_model, dataSet, script_args):
        """
        Create list of PMML network layers from Keras Model object.
        
        Parameters
        ----------
        keras_model : Keras model object
            Keras model object
        dataSet : String
            Name of the dataset
    
        Returns
        -------
        network_layers: Nyoka Object
            PMML network layer object 
        """
        network_layers = []
        model_layers = keras_model.layers
        first_layer = model_layers[0]
        if first_layer.__class__.__name__ != "InputLayer":
            input_layer = self._create_an_input_layer(first_layer, dataSet, script_args)
            if input_layer:
                network_layers.append(input_layer)
        for layer in model_layers:
            layer_type = layer.__class__.__name__
            net_layer = KerasNetworkLayer(layer,dataSet, layer_type,script_args)
            network_layers.append(net_layer)
        return network_layers

    def __init__(self, keras_model, model_name, dataSet=None, predictedClasses=None, script_args=None):
        if not model_name:
            model_namme = keras_model.name
        network_layers = self._create_layers(keras_model, dataSet, script_args)
        local_trans = None
        mining_schema = KerasMiningSchema(dataSet, predictedClasses)
        if dataSet == 'image' or script_args:
            local_trans = KerasLocalTransformations(keras_model, dataSet, script_args)
        function_Name = "classification" if predictedClasses else "regression"
        ny.DeepNetwork.__init__(self, modelName=model_name,
                                functionName=function_Name, algorithmName=None,
                                normalizationMethod="none", numberOfLayers=len(network_layers),
                                isScorable=True, Extension=None, MiningSchema=mining_schema,
                                Output=KerasOutput(predictedClasses), LocalTransformations=local_trans,
                                ModelStats=None, ModelExplanation=None, Targets=None,
                                NetworkLayer=network_layers, NeuralOutputs=None,
                                ModelVerification=None)


class KerasToPmml(ny.PMML):
    """
    KerasToPmml exports the Keras model object into PMML file using nyoka.
    
    Parameters
    ----------
    keras_model : Keras model object
        Keras model object 
    model_name : String
        Name to be given to the model in PMML.
    description : Sting (Optional)
        Description to be shown in PMML
    dataSet : String (Optional)
        Name of the dataset. Value is 'image' for Image Classifier, 'None' or any other value is for the rest. 
    predictedClasses : List
        List of the class names for which model has been trained. If not provided, assumed to be regression model.
    script_args : Dictionary
        Contains information of the script to be used to convert `image` data into base64 string. Required when dataSet=`image`.
        Required attributes - 
            content : string or function
                The content of the script
            def_name : string
                name of the function to be used. Required when content is string
            return_type : string
                The return type of the function. Valid values are ('string', 'double', 'float','integer')
            encode : boolean
                The representation of the script in PMML. If True, the script will be represented as base64 encoded string, else as plain text.
                If not provided, default value `True` is considered.

    Returns
    -------
    Creates PMML object, this can be saved in file using export function
    """

    @property
    def content_error(self):
        return "`content` should be present in script_args, which is either a function or a string (script content)"

    @property
    def def_name_error(self):
        return "`def_name`, the name of the funciton is required when `content` is a string."

    @property
    def ret_type_error(self):
        return "`return_type` of the preprocessing function is required. Valid return types are ('string', 'double', 'float', 'intger')"

    @property
    def ret_type_value_error(self):
        return "Valid return types are ('string', 'double', 'float', 'intger')"

    @property
    def encode_error(self):
        return "Valid values for `encode` are (True, False)"
    
    def validate_script_args(self, script_args):
        assert 'content' in script_args, self.content_error
        if script_args['content'].__class__.__name__ == 'str':
            assert 'def_name' in script_args, self.def_name_error
        assert 'return_type' in script_args, self.ret_type_error
        ret_type = script_args['return_type'].lower()
        assert ret_type in ("string", "double", "float", "intger"), self.ret_type_value_error
        if 'encode' in script_args:
            assert script_args['encode'] in [True, False], self.encode_error


    def __init__(self, keras_model, model_name=None, description=None,copyright=None,\
        dataSet=None, predictedClasses=None, script_args=None):
        if not dataSet:
            dataSet = 'input'
        data_dict = KerasDataDictionary(dataSet, predictedClasses, script_args)
        trans_dict = None
        if script_args:
            self.validate_script_args(script_args)
            trans_dict = KerasTransformationDictionary(dataSet,script_args)
        super(KerasToPmml, self).__init__(
            version="4.4", Header=KerasHeader(description=description, copyright=copyright),
            DataDictionary=data_dict, TransformationDictionary= trans_dict, DeepNetwork=[
                KerasNetwork(keras_model=keras_model, 
                model_name=model_name, 
                dataSet=dataSet, 
                predictedClasses=predictedClasses,
                script_args=script_args)])
