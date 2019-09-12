import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from keras import applications
from keras.layers import *
from keras.models import Model
from keras.models import Sequential

from keras_retinanet.models import load_model
from nyoka import RetinanetToPmml
from nyoka import PMML44 as pml
from nyoka.Base64 import FloatBase64
import unittest
import requests
import json
from requests.auth import HTTPBasicAuth
from adapaUtilities import AdapaUtility
from dataUtilities import DataUtility

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd

class TestCases(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("******* Unit Test for RetinaNet *******")
        url = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
        r = requests.get(url)

        with open('resnet50_coco_best_v2.1.0.h5', 'wb') as f:
            f.write(r.content)

        classes = json.load(open("nyoka/tests/categories_coco.json",'r'))
        self.classes = list(classes.values())
        self.adapa_utility = AdapaUtility()
        self.model = load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')


    def test_01(self):
        RetinanetToPmml(
            model=self.model,
            input_shape=(224,224,3),
            input_format='image',
            backbone_name='resnet',
            trained_classes=classes,
            pmml_file_name="RetinaNet.pmml"
        )
        model_name  = self.adapa_utility.upload_to_zserver('RetinaNet.pmml')
        z_boxes, z_scores, z_labels = self.adapa_utility.score_in_zserver(model_name, 'nyoka/tests/test_image_retinanet.png','RN')
        img = load_img('nyoka/tests/test_image_retinanet.png')
        img = img_to_array(img)
        img = preprocess_input(img)
        test = np.expand_dims(img, axis=0)
        boxes, scores, labels = self.model.predict(test)
        scores = scores.flatten()
        boxes = boxes.reshape(-1,4)
        labels = labels.flatten()
        
        scores_cnt = 0
        for a,b in zip(scores, z_scores):
            a = "{:.4f}".format(a)
            b = "{:.4f}".format(b)
            if a!=b:
                scores_cnt += 1

        labels_cnt = 0
        for a,b in zip(labels, z_labels):
            b = self.classes.index(b)
            if a!=b:
                labels_cnt += 1

        boxes_cnt = 0
        for a,b in zip(boxes, z_boxes):
            for a_,b_ in zip(a,b):
                a_ = "{:.2f}".format(a_)
                b_ = "{:.2f}".format(b_)
                if a_ != b_:
                    boxes_cnt += 1
        
        self.assertEqual(scores_cnt, 0)
        self.assertEqual(labels_cnt, 0)
        self.assertEqual(boxes_cnt, 0)

    @classmethod
    def tearDownClass(self):
        print("\n******* Finished *******\n")
     
if __name__ == '__main__':
    unittest.main(warnings='ignore')
        
        