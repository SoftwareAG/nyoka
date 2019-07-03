
from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from nyoka.keras.keras_model_to_pmml import KerasToPmml
from nyoka import PMML43Ext as pml

class MaskrcnnToPMML:
    def __init__(self, model, classes=None):
        self.keras_model = model.keras_model
        self.mode = model.mode
        self.config = model.config
        self.classes = classes
        self.pmml_obj = KerasToPmml(self.keras_model,dataSet='image',predictedClasses=self.classes)
        self.dump_config()

    def dump_config(self):
        config_vars = [ var for var in self.config.__dir__() if var.isupper()]
        config_dict = {}
        for var in config_vars:
            val=getattr(self.config,var)
            if val.__class__.__name__ == "ndarray":
                val = val.tolist()
            config_dict[var] = val
        info_dict = {}
        info_dict["config"] = config_dict
        info_dict["mode"] = self.mode
        exten_obj = pml.Extension(anytypeobjs_=[str(info_dict)], name="config")
        self.pmml_obj.DeepNetwork[0].Extension = [exten_obj]

    def export(self, fileName):
        self.pmml_obj.export(open(fileName,'w'),0)
