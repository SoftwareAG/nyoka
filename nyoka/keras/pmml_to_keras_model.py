import os,sys,ast
import numpy as np
from keras.models import Model
from keras.layers import *
import keras.layers as lays
from keras.preprocessing import image
from pprint import pprint

def update_progress(title, progress, status):
    barLength = 30
    block = int(round(barLength * progress))
    char = "#"
    bar = char * block + "-" * (barLength - block)
    text = "\r{0}: [{1}] {2}% {3}".format(title, bar, round(progress * 100), status)
    sys.stdout.write(text + " " * (100 - len(text)))
    sys.stdout.flush()
    if progress == 1: print()

class GenerateKerasModel:
    def __init__(self, pmml):
        self.nyoka_pmml = pmml
        self.image_input = None
        self.layer_input = None
        self.model = None
        self.layers_outputs = {}
        self.model = self._build_model()

    def _get_layer_weights(self, layer):
        """ get layer weights from nyokaBase pmml network layer"""
        layer_ws = layer.get_LayerWeights()
        weights = None
        if layer_ws:
            weights = layer_ws.weights()
        return weights

    def _get_layer_bias(self, layer):
        """ get layer bias from nyokaBase pmml network layer"""
        layer_bs = layer.get_LayerBias()
        bias = None
        if layer_bs:
            bias = layer_bs.weights()
        return bias

    def _get_layer_recurrent_weights(self, layer):
        """ get layer weights from nyokaBase pmml network layer"""
        layer_ws = layer.get_LayerRecurrentWeights()
        weights = None
        if layer_ws:
            weights = layer_ws.weights()
        return weights

    def _get_reshaped_layer_weights(self, layer_ws, out_shape, use_bs):
        """  takes flatten array of layer weights from nyokaBase pmml layer and reshaped it to
        out_shape to set as reconstructed model layer weights """
        reshape_layer_ws = []
        if len(out_shape) == 1:
            # print ('Cmae here for recurrent weights',out_shape[0])
            new_array = np.array(layer_ws).reshape(out_shape[0]).astype("float32")
            # print (new_array.shape)
            reshape_layer_ws.append(new_array)
        elif len(out_shape) == 2:
            if use_bs:
                # print ('Cmae here for Dense')
                new_array = np.array(layer_ws).reshape(out_shape[0]).astype("float32")
                reshape_layer_ws.append(new_array)
            else:
                start_indx = 0
                for item in out_shape:
                    stop_indx = start_indx + reduce(lambda x, y: x * y, item)
                    new_array = np.array(layer_ws)[start_indx:stop_indx].reshape(item).astype("float32")
                    reshape_layer_ws.append(new_array)
                    start_indx = stop_indx
        elif len(out_shape) == 3:
            # print ('Cmae3 here for weights',out_shape[0])
            new_array = np.array(layer_ws).reshape(out_shape[0]).astype("float32")
            reshape_layer_ws.append(new_array)
        else:
            reshape_layer_ws = np.array_split(layer_ws, len(out_shape))
        # print ('reshape_layer_ws >>>>',type(reshape_layer_ws))
        return reshape_layer_ws

    def _get_weights_shape(self, layer):
        """ get reconstructed model layer weights shape, so that we know how to reshape the
        flatten array to this shape"""
        ws_out_shape = []
        l_ws = layer.weights
        if l_ws:
            for item in l_ws:
                ws_out_shape.append(tuple(item.get_shape().as_list()))
        return ws_out_shape

    def _construct_layer(self, layer_input_names, layer_type, layer_name, kwargs):
        """ build model object by adding layers and to model obj."""
        x = None
        if len(layer_input_names) == 1:
            layer_input = self.layers_outputs[layer_input_names[0] + "_output"]
        else:
            layer_input = []
            for name in layer_input_names:
                layer_input.append(self.layers_outputs[name + "_output"])
        if layer_type == "Conv2D":
            if kwargs["activation_function"] =='rectifier':
                kwargs["activation_function"] = "relu"
            elif kwargs["activation_function"] =='tanch':
                kwargs["activation_function"] = "tanh"
            x = Conv2D(kwargs["filters"], kwargs["kernel"], padding=kwargs["pad"], strides=kwargs["stride"], use_bias=kwargs["use_bs"], dilation_rate=kwargs["dilation_rate"], activation=kwargs["activation_function"], name=layer_name)(layer_input)
        elif layer_type ==  "SeparableConv2D":
            x =  SeparableConv2D(kwargs["filters"], kwargs["kernel"], padding=kwargs["pad"], use_bias=kwargs["use_bs"], name=layer_name)(layer_input)
        elif layer_type == "BatchNormalization":
            x = BatchNormalization(name=layer_name, axis=kwargs["axis"], epsilon=kwargs["epsilon"], scale=kwargs["scale"])(layer_input)
        elif layer_type == "ReLU":
            x = ReLU(6., name=layer_name)(layer_input)
        elif layer_type == "Activation":
            if kwargs["activation_function"] == "relu6":
                x = ReLU(6., name=layer_name)(layer_input)
            elif kwargs["activation_function"] == "rectifier":
                max_val = float(kwargs["max_value"]) if kwargs["max_value"] else 1.0
                x = ReLU(max_val, name=layer_name)(layer_input)
            else:
                x = Activation(kwargs["activation_function"], name=layer_name)(layer_input)
        elif layer_type == "Dense":
            if kwargs["activation_function"] =='rectifier':
                kwargs["activation_function"] = "relu"
            elif kwargs["activation_function"] =='tanch':
                kwargs["activation_function"] = "tanh"
            x = Dense(int(kwargs["units"]), activation=kwargs["activation_function"], name=layer_name)(layer_input)
        elif layer_type == "MaxPooling2D":
            if not kwargs["stride"]:
                kwargs["stride"]=None
            x = MaxPooling2D(kwargs["poolSize"], strides=kwargs["stride"], padding=kwargs["pad"], name=layer_name)(layer_input)
        elif layer_type == "MergeLayer":
            if kwargs["merge_layer_op"] == "add":
                x = lays.add(layer_input, name=layer_name)
            if kwargs["merge_layer_op"] == "concatenate":
                x = lays.concatenate(layer_input, name=layer_name, axis=kwargs["merge_axis"])
        elif layer_type == "GlobalAveragePooling2D":
            x = GlobalAveragePooling2D(name=layer_name)(layer_input)
        elif layer_type == "GlobalMaxPooling2D":
            x = GlobalMaxPooling2D(name=layer_name)(layer_input)
        elif layer_type == "AveragePooling2D":
            x = AveragePooling2D(kwargs["stride"], name=layer_name)(layer_input)
        elif layer_type == "ZeroPadding2D":
            tempPad = kwargs['pad']
            if len(tempPad) ==4:
                kwargs['pad']=((tempPad[0],tempPad[1]),(tempPad[2],tempPad[3]))
                # print (kwargs['pad'])
                x = ZeroPadding2D(padding=kwargs["pad"], name=layer_name)(layer_input)
            else:
                x = ZeroPadding2D(padding=kwargs["pad"], name=layer_name)(layer_input)
        elif layer_type == "DepthwiseConv2D":
            x = DepthwiseConv2D(kwargs["kernel"], strides=kwargs["stride"], padding=kwargs["pad"], use_bias=kwargs["use_bs"], dilation_rate=kwargs["dilation_rate"], depth_multiplier=kwargs["depth_multiplier"], name=layer_name)(layer_input)
        elif layer_type == "Reshape":
            x = Reshape(kwargs["output_shape"], name=layer_name)(layer_input)
        elif layer_type == "Dropout":
            x = Dropout(rate=kwargs["dropout_rate"], name=layer_name)(layer_input)
        elif layer_type == "Flatten":
            x = Flatten(name=layer_name)(layer_input)
        elif layer_type == "LSTM":
            if 'return_sequences' in kwargs:
                x= LSTM(int(kwargs["units"]), return_sequences =  kwargs["return_sequences"], name=layer_name)(layer_input)
            else:
                x= LSTM(int(kwargs["units"]), name=layer_name)(layer_input)
        return x

    def _add_layer_weights(self):
        """ collect layer weights from nyokaBase pmml file and add them to model layers """
        pmml_deep_net = self.nyoka_pmml.DeepNetwork[0]
        pmml_net_layers = pmml_deep_net.NetworkLayer
        if len(pmml_net_layers) == len(self.model.layers):
            for i in range(len(pmml_net_layers)):
                progress = float(i + 1) / float(len(pmml_net_layers))
                status = pmml_net_layers[i].get_layerId()
                update_progress("Applying Weights", progress, status)
                nyoka_layer = pmml_net_layers[i]
                layer_ws = self._get_layer_weights(nyoka_layer)
                layer_bs = self._get_layer_bias(nyoka_layer)
                layer_rec_ws=self._get_layer_recurrent_weights(nyoka_layer)

                use_bs = True if layer_bs else False
                if layer_ws:
                    out_shapes_as_lst = self._get_weights_shape(self.model.layers[i])
                    rs_layer_ws = self._get_reshaped_layer_weights(layer_ws, out_shapes_as_lst, use_bs)
                    if layer_rec_ws:
                        rs_layer_ws.append(self._get_reshaped_layer_weights(layer_rec_ws, [out_shapes_as_lst[1]], use_bs)[0])
                    if layer_bs:
                        rs_layer_ws.append(np.array(layer_bs).astype("float32"))
                    if rs_layer_ws:
                        self.model.layers[i].set_weights(rs_layer_ws)
        else:
            print("nyoka pmml layers and reconstructed model layers not same in length,"
                  "please check")

    def _build_model(self):
        """ build keras model object from nyokaBase pmml object"""
        layer_output = None
        pmml_deep_net = self.nyoka_pmml.DeepNetwork[0]
        pmml_net_layers = pmml_deep_net.NetworkLayer
        for indx, pmml_layer in enumerate(pmml_net_layers):
            kwargs = {}
            layer_type = pmml_layer.get_layerType()
            layer_name = pmml_layer.get_layerId()
            connection_layer_names = pmml_layer.get_connectionLayerId()
            layer_params = pmml_layer.get_LayerParameters()
            input_shape = eval(layer_params.get_inputDimension())
            kwargs["use_bs"] = True if self._get_layer_bias(pmml_layer) else False
            kwargs["output_shape"] = eval(layer_params.get_outputDimension())
            kwargs["merge_layer_op"] = layer_params.get_mergeLayerOp()
            kwargs["axis"] = layer_params.get_batchNormalizationAxis()
            kwargs["scale"] = layer_params.get_batchNormalizationScale()
            kwargs["epsilon"] = layer_params.get_batchNormalizationEpsilon()
            kwargs["depth_multiplier"] = layer_params.get_depthMultiplier()
            kwargs["dropout_rate"] = layer_params.get_dropoutRate()
            kwargs["max_value"] = layer_params.get_max_value()
            kwargs["units"] = layer_params.get_units()
            try:
                kwargs["pad"] = eval(str(layer_params.get_paddingDims()))
            except:
                kwargs["pad"] = str(layer_params.get_paddingDims())
            kwargs["pad"] = str(layer_params.get_paddingType()) if kwargs["pad"] is None else kwargs["pad"]
            if layer_params.get_kernel():
                kwargs["kernel"] = eval(layer_params.get_kernel())
            if layer_params.get_stride():
                kwargs["stride"] = eval(layer_params.get_stride())
            if layer_params.get_featureMaps():
                kwargs["filters"] = int(layer_params.get_featureMaps())
            if layer_params.get_activationFunction():
                kwargs["activation_function"] = layer_params.get_activationFunction().lower()
            if layer_params.get_poolSize():
                kwargs["poolSize"] = eval(layer_params.get_poolSize())
            if layer_params.get_mergeLayerConcatOperationAxes():
                kwargs["merge_axis"] = int(layer_params.get_mergeLayerConcatOperationAxes())
            if layer_params.get_dilationRate():
                kwargs["dilation_rate"] = eval(layer_params.get_dilationRate())
            if layer_params.get_return_sequences():
                kwargs['return_sequences'] = layer_params.get_return_sequences()
            progress = float(indx + 1) / float(len(pmml_net_layers))
            update_progress("Creating Layers", progress, layer_name)
            if connection_layer_names:
                layer_input_names = connection_layer_names.split(", ")
            else:
                layer_input_names = []
            if layer_type == "Input":
                if (len(input_shape)==2) & (input_shape[1]==1):
                    # print ('input_shape>>>>>',input_shape)
                    layer_output = Input(shape=(input_shape[0],), name=layer_name)
                    self.layers_outputs[layer_name+"_output"] = layer_output
                    self.image_input = layer_output
                else:
                    layer_output = Input(shape=input_shape, name=layer_name)
                    self.layers_outputs[layer_name+"_output"] = layer_output
                    self.image_input = layer_output
            else:
                
                layer_output = self._construct_layer(layer_input_names, layer_type, layer_name, kwargs)
                self.layers_outputs[layer_name+"_output"] = layer_output

        # Create model.
        self.model = Model(self.image_input, layer_output, name=pmml_deep_net.get_modelName())
        self._add_layer_weights()
        return self.model