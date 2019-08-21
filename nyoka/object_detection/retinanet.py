from __future__ import absolute_import

import sys
import os
import warnings
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
warnings.formatwarning = lambda msg, *args, **kwargs: str(msg)+'\n'

import numpy as np
from nyoka import PMML44 as pml

from nyoka.keras import keras_model_to_pmml as kerasAPI

class RetinanetToPmml:
    
    """
    Write a PMML file for RetinaNet model.

    Parameters:
    -----------
    model: 
        RetinaNet model object
    input_shape: tuple 
        Shape of each training image
    input_data: string (optional. default='image')
        Input format to be used during inference with the PMML. Valid values are - 
            "image" : Original image in png format
            "encoded" : Base64 encoded string of the image
    trained_classes : list or tuple
        List of the classes on which the model was trained. If not provided, `max_detections`(1 to 300) classes will be used
    pmml_file_name: string
        Name of the PMML file
    
    """

    def __init__(self, model, input_shape, input_data="image", trained_classes=None, pmml_file_name="from_retinanet.pmml"):
        assert model.layers[-1].__class__.__name__ == 'FilterDetections', 'Given model is not an inference model!'
        assert input_data in ['image','encoded'], "Invalid input_data type. Valid values are `['image', 'encoded']`"
        self._pyramid_layers = ("P3", "P4", "P5", "P6", "P7")
        self.model = model
        self.input_shape = input_shape
        self.pmml_obj = None
        self._layer_outputs = dict()
        self.generate_pmml(model, input_shape,input_data,trained_classes,pmml_file_name)
        self.pmml_obj.export(open(pmml_file_name,'w'),0)

    def generate_backbone_anchor(self, model, input_shape, input_data, trained_classes):
        from keras.models import Sequential
        from keras import backend as K
        sample_data = np.random.random(size=input_shape)
        nan_index = np.isnan(sample_data)
        sample_data[nan_index] = 0.5
        test = np.expand_dims(sample_data, axis=0)
        layers = []
        for l in model.layers:
            if l.__class__.__name__ == "Model":
                break
            layers.append(l)
        inp = model.input
        outputs_tens = [layer.output for layer in layers[1:]] 
        functor = K.function([inp], outputs_tens )
        outputs_tens.insert(0,inp)
        layer_outs = functor([test, 1.])
        layer_outs.insert(0, test)
            
        self._layer_outputs = {}
        for lay, out in zip(layers, layer_outs):
            self._layer_outputs[lay.name] = out

        mod = Sequential()
        for l in layers[1:]:
            mod.add(l)
        
        if trained_classes == None:
            warnings.warn("trained_classes are not provided. Default `max_classes`(1 to 300) will be considered.")
            trained_classes = ["Category_"+str(i+1).zfill(3) for i in range(300)]

        group1_pmml = kerasAPI.KerasToPmml(mod,model_name=model.name,dataSet=input_data, predictedClasses=trained_classes)
        for idx, layer in enumerate(group1_pmml.DeepNetwork[0].NetworkLayer):
            if idx==0:
                input_shape = output_shape = str(self._layer_outputs[layer.layerId].shape[1:])
            else:
                connected_layers = layer.connectionLayerId.split(", ")
                if len(connected_layers) > 1:
                    input_shape = []
                    for con_lay in connected_layers:
                        input_shape.append(str(self._layer_outputs[con_lay].shape[1:]))
                    input_shape = ", ".join(input_shape)
                else:
                    input_shape = str(self._layer_outputs[connected_layers[0]].shape[1:])
                output_shape = str(self._layer_outputs[layer.layerId].shape[1:])
            layer.LayerParameters.inputDimension = input_shape
            layer.LayerParameters.outputDimension = output_shape

        return group1_pmml

    def generate_submodel(self, model):
        from keras import backend as K
        layer_output_ = dict()
        for lay in self._pyramid_layers:
            inp = model.get_input_at(0)
            outputs_tens_ = [layer.output for layer in model.layers[1:]]
            functor_ = K.function([inp], outputs_tens_ )
            test_ = self._layer_outputs[lay]
            layer_outs_ = functor_([test_, 1.])
            layer_outs_.insert(0, test_)
            layer_output_[lay] = layer_outs_

        net_layers_group=list()
        for idx, name in enumerate(self._pyramid_layers):
            nyoka_pmml_reg_mod = kerasAPI.KerasToPmml(model)
            del nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer[0]
            nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer[0].connectionLayerId = name
            for idx_, lay in enumerate(nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer):
                lay.LayerParameters.inputDimension = str(tuple(layer_output_[name][idx_].shape[1:]))
                lay.LayerParameters.outputDimension = str(tuple(layer_output_[name][idx_+1].shape[1:]))
                
                lay.layerId = lay.layerId+"_"+name
                if idx_ != 0:
                    lay.connectionLayerId = lay.connectionLayerId+"_"+name
            last_id = nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer[-1].layerId
            self._layer_outputs[last_id] = layer_output_[name][-1]
            net_layers_group.extend(nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer)
        return net_layers_group

    def generate_inference_layers(self, model, input_shape):
        from keras import backend as K
        from keras import Model
        sample_data = np.random.random(size=input_shape)
        nan_index = np.isnan(sample_data)
        sample_data[nan_index] = 0.5
        test = np.expand_dims(sample_data, axis=0)
        inference_layers= [lay for lay in model.layers[-8:] if lay.__class__.__name__ != "Model"]
        inference_layers_outputs = dict()
        for lay in inference_layers:
            layer_name = lay.name
            intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            inference_layers_outputs[layer_name] = intermediate_layer_model.predict(test)


        inference_network_layers = list()
        for lay in inference_layers:
            
            connectLayerIds=list()
            for idx,lay_ in enumerate(lay._inbound_nodes[0].inbound_layers):
                if hasattr(lay_,'layers'):
                    name = lay_.layers[-1].name+"_"+self._pyramid_layers[idx]
                else:
                    name = lay_.name
                connectLayerIds.append(name)
            if lay.__class__.__name__ == 'FilterDetections':
                connectLayerIds = connectLayerIds[:2]
            inputDimesion = list()
            for id_ in connectLayerIds:
                if id_ in self._layer_outputs.keys():
                    inputDimesion.append(str(self._layer_outputs[id_].shape[1:]))
                else:
                    inputDimesion.append(str(inference_layers_outputs[id_].shape[1:]))
            
            inputDimesion = ", ".join(inputDimesion)

            network_layer=kerasAPI.KerasNetworkLayer(lay,"dataSet",lay.__class__.__name__, connectLayerIds)
            network_layer.connectionLayerId = ", ".join(connectLayerIds)
            network_layer.LayerParameters.inputDimension = inputDimesion
            if lay.name in list(self._layer_outputs.keys()):
                network_layer.LayerParameters.outputDimension = str(tuple(self._layer_outputs[lay.name].shape[1:]))
            else:
                if lay.__class__.__name__ == 'FilterDetections':
                    new_shape_lst = [0,0]
                    for o_shape in inference_layers_outputs[lay.name]:
                        o_shape = o_shape.shape[1:]
                        if len(o_shape) == 1:
                            shp = (o_shape[0], 1)
                        else:
                            shp = o_shape
                        new_shape_lst[0] = shp[0]
                        new_shape_lst[1] += shp[1]
                    outputDimesnion = str(tuple(new_shape_lst))
                else:
                    outputDimesnion = str(inference_layers_outputs[lay.name].shape[1:])
                network_layer.LayerParameters.outputDimension = outputDimesnion
            inference_network_layers.append(network_layer)

        return inference_network_layers

    def get_output(self):
        out_flds = []
        out_flds.append(
            pml.OutputField(
                name="boxes_scores_labels_json",
                dataType="string",
                feature="array"
            )
        )
        out_flds.append(
            pml.OutputField(
                name="labels",
                dataType="string",
                feature="predictedValue"
            )
        )
        return pml.Output(OutputField=out_flds)

    @property
    def description(self):
        return 'RetinaNet model in PMML'

    @property
    def algorith_name(self):
        return 'RetinaNet'


    def generate_pmml(self, model, input_shape, input_data, trained_classes, pmml_file_name):
    
        group_1_pmml_obj = self.generate_backbone_anchor(model, input_shape, input_data, trained_classes)
        regression_submodel_layers = self.generate_submodel(model.layers[-8])
        classification_submodel_layers = self.generate_submodel(model.layers[-4])
        inference_layers = self.generate_inference_layers(model, input_shape)
        group_1_pmml_obj.DeepNetwork[0].NetworkLayer.extend(
            regression_submodel_layers+classification_submodel_layers+inference_layers
            )
        group_1_pmml_obj.DeepNetwork[0].numberOfLayers = len(group_1_pmml_obj.DeepNetwork[0].NetworkLayer)
        group_1_pmml_obj.DeepNetwork[0].algorithmName = self.algorith_name
        group_1_pmml_obj.DeepNetwork[0].Output = self.get_output()
        group_1_pmml_obj.Header.description=self.description
        self.pmml_obj = group_1_pmml_obj

        