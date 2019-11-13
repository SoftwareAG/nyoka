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

    Parameters
    -----------
    model : 
        RetinaNet model object
    input_shape : tuple 
        Expected shape of the images to be scored
    backbone_name : string
        Name of backbone used to train the model. Valid values are `['resnet', 'mobilenet', 'densenet', 'vgg']`
    input_format : string (optional. default='image')
        Input format to be used during inference with the PMML. Valid values are - 
            "image" : Original image in png format
            "encoded" : Base64 encoded string of the image
    trained_classes : list or tuple
        List of the classes on which the model was trained. If not provided, default(1 to 80) classes will be used
    pmml_file_name : string (default='from_retinanet.pmml')
        Name of the PMML file
    script_args : Dictionary or None
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
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description of the model

    Returns
    -------
    Creates Nyoka's PMML object and exports it to `pmml_file_name`
    
    """

    def inference_error(self):
        return "Given model is not an inference model!"

    def input_format_error(self):
        return "Invalid input_format type. Valid values are `['image', 'encoded']`"

    def backbone_name_error(self):
        return "Invalid backbone_name. Valid values are `['resnet', 'mobilenet', 'densenet', 'vgg']`"

    def __init__(self, model, input_shape, backbone_name, input_format="image", trained_classes=None,
     pmml_file_name="from_retinanet.pmml", script_args=None, model_name=None, description=None):
        assert model.layers[-1].__class__.__name__ == 'FilterDetections', self.inference_error()
        assert input_format in ['image','encoded'], self.input_format_error()
        assert backbone_name in ['resnet', 'mobilenet', 'densenet', 'vgg'], self.backbone_name_error()

        self.backbone_name = backbone_name
        self.model = model
        self.input_shape = input_shape
        self.input_format = input_format
        self.script_args = script_args
        self.model_name = model_name
        self.description = description if description else "RetinaNet model in PMML"

        self.pmml_obj = None
        self._pyramid_layers = ("P3", "P4", "P5", "P6", "P7")
        self._layer_outputs = dict()

        self.generate_pmml(model, input_shape,input_format,trained_classes)
        self.pmml_obj.export(open(pmml_file_name,'w'),0)


    def generate_beckbone_anchors(self, model, input_format, trained_classes):
        """
        Generates PMML object for the backbone + anchors

        Parameters
        ----------
        model : 
            RetinaNet model object
        input_format : string
            Input format to be used during inference with the PMML. Valid values are - 
                "image" : Original image in png format
                "encoded" : Base64 encoded string of the image
        trained_classes : List
            List of class names for which the model was trained

        Returns
        -------
        Nyoka's PMML object

        """
        from keras.models import Sequential
        mod = Sequential()
        for l in model.layers[1:]:
            if l.__class__.__name__ == "Model":
                break
            mod.add(l)
        if trained_classes == None:
            warnings.warn(f"trained_classes are not provided. Maximum 80 classes will be considered.")
            trained_classes = ["Category_"+str(i+1).zfill(2) for i in range(80)]

        group1_pmml = kerasAPI.KerasToPmml(mod,model_name=self.model_name if self.model_name else\
             "KerasRetinaNet"+self.input_format.title(),dataSet=input_format, description=self.description,
         predictedClasses=trained_classes, script_args=self.script_args)
        return group1_pmml

    
    def generate_submodel(self, submodel):
        """
        Generates multiple PMML object for the regression and classification submodel of RetinaNet for each connected pyramid layers

        Parameters
        ----------
        submodel :
            The Regression or the Classification submodel

        Returns
        -------
        List of Nyoka's NetworkLayer object for all the submodels
        """
        net_layers_group=list()
        for idx, name in enumerate(self._pyramid_layers):
            nyoka_pmml_reg_mod = kerasAPI.KerasToPmml(submodel)
            del nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer[0]
            nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer[0].connectionLayerId = name
            for idx_, lay in enumerate(nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer):                
                lay.layerId = lay.layerId+"_"+name
                if idx_ != 0:
                    lay.connectionLayerId = lay.connectionLayerId+"_"+name
            net_layers_group.extend(nyoka_pmml_reg_mod.DeepNetwork[0].NetworkLayer)
        return net_layers_group

    
    def generate_inference_layers(self, model):
        """
        Generates PMML object for the inference layers of RetinaNet

        Parameters
        ----------
        model :
            RetinaNet model object

        Returns
        -------
        List of Nyoka's NetworkLayer
        """
        inference_layers= [lay for lay in model.layers[-8:] if lay.__class__.__name__ != "Model"]
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
            network_layer=kerasAPI.KerasNetworkLayer(lay,"dataSet",lay.__class__.__name__, connectLayerIds)
            network_layer.connectionLayerId = ", ".join(connectLayerIds)
            inference_network_layers.append(network_layer)
        return inference_network_layers

    
    def assign_shapes(self, model, input_shape, pmml_without_shape):
        """
        Assigns the shape information to each NetworkLayer of the PMML

        Parameters
        ----------
        model :
            RetinaNet model object
        input_shape : tuple
            Input shape of each image used during training
        pmml_without_shape :
            Generated PMML object without shape information

        Returns
        -------
        Nyoka's PMML object
        """
        from keras.models import Sequential
        from keras import backend as K
        from keras import Model

        layer_output_dict = dict()

        # dummy data for shape calculation
        sample_data = np.random.random(size=input_shape)
        nan_index = np.isnan(sample_data)
        sample_data[nan_index] = 0.5
        test = np.expand_dims(sample_data, axis=0)
        
        # backbone and anchors
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
        
        for lay, out in zip(layers, layer_outs):
            layer_output_dict[lay.name] = out

        # regression submodel
        regression_submodel = model.layers[-8]
        for lay in self._pyramid_layers:
            inp = regression_submodel.get_input_at(0)
            outputs_tens_ = [lay_.output for lay_ in regression_submodel.layers[1:]]
            functor_ = K.function([inp], outputs_tens_ )
            test_ = layer_output_dict[lay]
            layer_outs_ = functor_([test_, 1.])
            for lay_in, lay_out in zip(regression_submodel.layers[1:], layer_outs_):
                layer_output_dict[lay_in.name+"_"+lay] = lay_out

        # classification submodel
        classification_submodel = model.layers[-4]
        for lay in self._pyramid_layers:
            inp = classification_submodel.get_input_at(0)
            outputs_tens_ = [lay_.output for lay_ in classification_submodel.layers[1:]]
            functor_ = K.function([inp], outputs_tens_ )
            test_ = layer_output_dict[lay]
            layer_outs_ = functor_([test_, 1.])
            for lay_in, lay_out in zip(classification_submodel.layers[1:], layer_outs_):
                layer_output_dict[lay_in.name+"_"+lay] = lay_out

        # inference layers
        inference_layers= [lay for lay in model.layers[-8:] if lay.__class__.__name__ != "Model"]
        for lay in inference_layers:
            layer_name = lay.name
            intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            layer_output_dict[layer_name] = intermediate_layer_model.predict(test)

        # assign shapes
        for net_layer in pmml_without_shape.DeepNetwork[0].NetworkLayer:
            input_shape = None
            if net_layer.connectionLayerId == "na":
                input_shape = output_shape = str(layer_output_dict[net_layer.layerId].shape[1:])
            else:
                connected_layers = net_layer.connectionLayerId.split(", ")
                if len(connected_layers) > 1:
                    input_shape = []
                    for con_lay in connected_layers:
                        input_shape.append(str(layer_output_dict[con_lay].shape[1:]))
                    input_shape = ", ".join(input_shape)
                else:
                    input_shape = str(layer_output_dict[connected_layers[0]].shape[1:])
                if net_layer.layerType == 'FilterDetections':
                    new_shape_lst = [0,0]
                    for o_shape in layer_output_dict[net_layer.layerId]:
                        o_shape = o_shape.shape[1:]
                        if len(o_shape) == 1:
                            shp = (o_shape[0], 1)
                        else:
                            shp = o_shape
                        new_shape_lst[0] = shp[0]
                        new_shape_lst[1] += shp[1]
                    output_shape = str(tuple(new_shape_lst))
                else:
                    output_shape = str(layer_output_dict[net_layer.layerId].shape[1:])
            net_layer.LayerParameters.inputDimension = input_shape
            net_layer.LayerParameters.outputDimension = output_shape
        return pmml_without_shape

    def get_output(self):
        """
        Generates Output for RetinaNet

        Returns
        -------
        Nyoka's Output object
        """
        out_flds = []
        out_flds.append(
            pml.OutputField(
                name="predicted_LabelBoxScore",
                dataType="string",
                feature="predictedValue",
                Extension = [pml.Extension(extender="ADAPA", name="format", value="JSON")]
            )
        )
        return pml.Output(OutputField=out_flds)

    def get_training_parameter(self):
        """
        Generates TrainingParameters for RetinaNet

        Returns
        -------
        Nyoka's TrainingParameters object
        """
        train_param = pml.TrainingParameters(architectureName='retinanet')
        return train_param

    def get_local_transformation(self):
        """
        Generates Trasformation information for RetinaNet

        Returns
        -------
        Nyoka's LocalTransformations object
        """
        apply = pml.Apply(
            function='KerasRetinaNet:getBase64StringFromBufferedInput',
            FieldRef = [pml.FieldRef(field=self.input_format)],
            Constant = [pml.Constant(valueOf_='tf' if self.backbone_name in ['mobilenet', 'densenet'] else 'caffe')]
        )
        der_fld = pml.DerivedField(
            name="base64String",
            optype="categorical",
            dataType="string",
            Apply = apply
        )
        return pml.LocalTransformations(DerivedField = [der_fld])

    
    def generate_pmml(self,model,input_shape,input_format,trained_classes):
        """
        Generates PMML object for RetinaNet by combining all different part's PMML object

        Parameters
        ----------
        model : 
            RetinaNet model object
        input_shape : tuple 
            Shape of each training image
        input_format : string (optional. default='image')
            Input format to be used during inference with the PMML. Valid values are - 
                "image" : Original image in png format
                "encoded" : Base64 encoded string of the image
        trained_classes : list or tuple
            List of the classes on which the model was trained. If not provided, default(1 to 80) classes will be used
            
        Returns
        -------
        Generated nyoka's PMML object
        """

        backbone_and_anchor = self.generate_beckbone_anchors(model, input_format, trained_classes)
        regression_submodel_layers = self.generate_submodel(model.layers[-8])
        classification_submodel_layers = self.generate_submodel(model.layers[-4])
        inference_layers = self.generate_inference_layers(model)
        backbone_and_anchor.DeepNetwork[0].NetworkLayer.extend(
            regression_submodel_layers+classification_submodel_layers+inference_layers
        )
        model_with_shape_info = self.assign_shapes(model, input_shape, backbone_and_anchor)
        model_with_shape_info.DeepNetwork[0].numberOfLayers = len(model_with_shape_info.DeepNetwork[0].NetworkLayer)
        model_with_shape_info.DeepNetwork[0].Output = self.get_output()
        model_with_shape_info.DeepNetwork[0].TrainingParameters = self.get_training_parameter()
        if self.input_format == 'image' and not self.script_args:
            model_with_shape_info.DeepNetwork[0].LocalTransformations = self.get_local_transformation() 
        model_with_shape_info.Header.description=self.description
        self.pmml_obj = model_with_shape_info
