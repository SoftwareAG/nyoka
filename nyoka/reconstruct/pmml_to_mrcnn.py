from nyoka.reconstruct import model as modellib
from nyoka.reconstruct import config
import sys,os
import numpy as np
import functools

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class MyConfig(config.Config):
    pass

def update_progress(title, progress, status):
    barLength = 30
    block = int(round(barLength * progress))
    char = "#"
    bar = char * block + "-" * (barLength - block)
    text = "\r{0}: [{1}] {2}% {3}".format(title, bar, round(progress * 100), status)
    sys.stdout.write(text + " " * (100 - len(text)))
    sys.stdout.flush()
    if progress == 1: print()


class GenerateMaskRcnnModel:

    def __init__(self, pmml):
        self.pmml = pmml
        self.config = None
        self.mode = None
        self.model = None
        self.LAYERID_INDEX_MAP = {}
        self.set_mode_and_config(pmml)
        self.create_layerId_index_dict()
        self.build_model()

    def set_mode_and_config(self, pmml):
        mnc = eval(pmml.DeepNetwork[0].Extension[0].anytypeobjs_[0])
        self.mode = mnc['mode']
        self.config = MyConfig()
        for key,val in mnc['config'].items():
            setattr(self.config,key,val)

    def create_layerId_index_dict(self):
        netLayers = self.pmml.DeepNetwork[0].NetworkLayer
        for index, layer in enumerate(netLayers):
            self.LAYERID_INDEX_MAP[layer.layerId] = index

    def _get_layer_weights(self, layer):
        """ get layer weights from nyoka pmml network layer"""
        layer_ws = layer.get_LayerWeights()
        weights = None
        if layer_ws:
            weights = layer_ws.weights()
        return weights

    def _get_layer_bias(self, layer):
        """ get layer bias from nyoka pmml network layer"""
        layer_bs = layer.get_LayerBias()
        bias = None
        if layer_bs:
            bias = layer_bs.weights()
        return bias

    def _get_reshaped_layer_weights(self, layer_ws, out_shape, use_bs):
        """  takes flatten array of layer weights from nyoka pmml layer and reshaped it to
        out_shape to set as reconstructed model layer weights """
        reshape_layer_ws = []
        if len(out_shape) == 1:
            new_array = np.array(layer_ws).reshape(out_shape[0]).astype("float32")
            reshape_layer_ws.append(new_array)
        elif len(out_shape) == 2:
            if use_bs:
                new_array = np.array(layer_ws).reshape(out_shape[0]).astype("float32")
                reshape_layer_ws.append(new_array)
            else:
                start_indx = 0
                for item in out_shape:
                    stop_indx = start_indx + functools.reduce(lambda x, y: x * y, item)
                    new_array = np.array(layer_ws)[start_indx:stop_indx].reshape(item).astype("float32")
                    reshape_layer_ws.append(new_array)
                    start_indx = stop_indx
        else:
            # layer_ws = np.array(layer_ws).astype("float32")
            reshape_layer_ws = np.array_split(layer_ws, len(out_shape))
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

    def _add_layer_weights(self):
        pmml_deep_net = self.pmml.DeepNetwork[0]
        pmml_net_layers = pmml_deep_net.NetworkLayer
        for index,layer in enumerate(self.model.keras_model.layers):
            progress = float(index + 1) / float(len(self.model.keras_model.layers))
            status = layer.name
            update_progress("Applying Weights", progress, status)

            if hasattr(layer,'layers'):
                for idx,lay in enumerate(layer.layers):
                    nyoka_layer = pmml_net_layers[self.LAYERID_INDEX_MAP[lay.name]]
                    layer_ws = self._get_layer_weights(nyoka_layer)
                    layer_bs = self._get_layer_bias(nyoka_layer)
                    use_bs = True if layer_bs else False
                    if layer_ws:
                        out_shapes_as_lst = self._get_weights_shape(self.model.keras_model.layers[index].layers[idx])
                        rs_layer_ws = self._get_reshaped_layer_weights(layer_ws, out_shapes_as_lst, use_bs)
                        if layer_bs:
                            rs_layer_ws.append(np.array(layer_bs).astype("float32"))
                        if rs_layer_ws:
                            self.model.keras_model.layers[index].layers[idx].set_weights(rs_layer_ws)
            else:
                nyoka_layer = pmml_net_layers[self.LAYERID_INDEX_MAP[status]]
                layer_ws = self._get_layer_weights(nyoka_layer)
                layer_bs = self._get_layer_bias(nyoka_layer)
                use_bs = True if layer_bs else False
                if layer_ws:
                    out_shapes_as_lst = self._get_weights_shape(self.model.keras_model.layers[index])
                    rs_layer_ws = self._get_reshaped_layer_weights(layer_ws, out_shapes_as_lst, use_bs)
                    if layer_bs:
                        rs_layer_ws.append(np.array(layer_bs).astype("float32"))
                    if rs_layer_ws:
                        self.model.keras_model.layers[index].set_weights(rs_layer_ws)
        
        
    def build_model(self):
        from keras import backend as K
        tf_session = K.get_session()
        with tf_session.as_default():
            self.model = modellib.MaskRCNN(mode=self.mode, config=self.config, model_dir=MODEL_DIR)
        
        self._add_layer_weights()
