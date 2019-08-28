
import sys,os
import requests
import unittest
from keras_retinanet.models import load_model
from nyoka import RetinanetToPmml
from nyoka import PMML44 as pml


class TestMethods(unittest.TestCase):

    
    def test_01(self):
        
        url = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
        r = requests.get(url)

        with open('resnet50_coco_best_v2.1.0.h5', 'wb') as f:
            f.write(r.content)
        model = load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')
        backbone = 'resnet'
        RetinanetToPmml(
            model,
            input_shape=(224,224,3),
            backbone_name=backbone,
            pmml_file_name="retinanet_with_coco.pmml"
        )
        recon_pmml_obj = pml.parse("retinanet_with_coco.pmml",True)
        binary_buffered = recon_pmml_obj.DataDictionary.DataField[0].Extension[0].value
        self.assertEqual(binary_buffered,'true')
        function = recon_pmml_obj.DeepNetwork[0].LocalTransformations.DerivedField[0].Apply.function
        self.assertEqual(function,'KerasRetinaNet:getBase64StringFromBufferedInput')
        scaling = recon_pmml_obj.DeepNetwork[0].LocalTransformations.DerivedField[0].Apply.Constant[0].valueOf_
        self.assertEqual(scaling,'caffe')


if __name__=='__main__':
    unittest.main(warnings='ignore')







