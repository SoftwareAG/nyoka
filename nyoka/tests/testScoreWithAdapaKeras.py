import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from keras import applications
from keras.layers import *
from keras.models import Model
from keras.models import Sequential

from nyoka import KerasToPmml
from nyoka import PMML44 as pml
from nyoka.Base64 import FloatBase64
import unittest
import base64
import requests
import json
from requests.auth import HTTPBasicAuth
from adapaUtilities import AdapaUtility
from dataUtilities import DataUtility

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd

class TestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("******* Unit Test for Keras *******")
        cls.adapa_utility = AdapaUtility()
        cls.data_utility = DataUtility()
        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
        activType='sigmoid'
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation=activType)(x)
        cls.model_final = Model(inputs =model.input, outputs = predictions,name='predictions')


    def test_01_image_classifier_with_image_as_input(self):
        
        cnn_pmml = KerasToPmml(self.model_final,model_name="MobileNetImage",description="Demo",\
            copyright="Internal User",dataSet='image',predictedClasses=['dogs','cats'])
        cnn_pmml.export(open('2classMBNet.pmml', "w"), 0)

        img = image.load_img('nyoka/tests/resizedCat.png')
        img = img_to_array(img)
        img = preprocess_input(img)
        imgtf = np.expand_dims(img, axis=0)
        model_pred=self.model_final.predict(imgtf)
        model_preds = {'dogs':model_pred[0][0],'cats':model_pred[0][1]}

        model_name  = self.adapa_utility.upload_to_zserver('2classMBNet.pmml')

        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, 'nyoka/tests/resizedCat.png','DN')
  
        self.assertEqual(abs(probabilities['cats'] - model_preds['cats']) < 0.00001, True)
        self.assertEqual(abs(probabilities['dogs'] - model_preds['dogs']) < 0.00001, True)

    def test_02_image_classifier_with_base64string_as_input(self):
        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (80, 80,3))
        activType='sigmoid'
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation=activType)(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
        
        cnn_pmml = KerasToPmml(model_final,model_name="MobileNetBase64",description="Demo",\
            copyright="Internal User",dataSet='imageBase64',predictedClasses=['dogs','cats'])
        cnn_pmml.export(open('2classMBNetBase64.pmml', "w"), 0)

        img = image.load_img('nyoka/tests/resizedTiger.png')
        img = img_to_array(img)
        img = preprocess_input(img)
        imgtf = np.expand_dims(img, axis=0)

        base64string = "data:float32;base64," + FloatBase64.from_floatArray(img.flatten(),12)
        base64string = base64string.replace("\n", "")
        csvContent = "imageBase64\n\"" + base64string + "\""
        text_file = open("input.csv", "w")
        text_file.write(csvContent)
        text_file.close()

        model_pred=model_final.predict(imgtf)
        model_preds = {'dogs':model_pred[0][0],'cats':model_pred[0][1]}

        model_name  = self.adapa_utility.upload_to_zserver('2classMBNetBase64.pmml')

        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, 'input.csv','DN')
  
        self.assertEqual(abs(probabilities['cats'] - model_preds['cats']) < 0.00001, True)
        self.assertEqual(abs(probabilities['dogs'] - model_preds['dogs']) < 0.00001, True)

    @unittest.skip("")
    def test_03_encoded_script(self):
        script_content = open("nyoka/tests/preprocess.py",'r').read()
        pmml_obj=KerasToPmml(self.model_final,
                    dataSet='image',
                    predictedClasses=['cat','dog'],
                    script_args = {
                        "content" : script_content,
                        "def_name" : "getBase64EncodedString",
                        "return_type" : "string",
                        "encode":True
                    }
                )
        pmml_obj.export(open("script_with_keras.pmml",'w'),0)
        self.assertEqual(os.path.isfile("script_with_keras.pmml"),True)
        reconPmmlObj = pml.parse("script_with_keras.pmml",True)
        content=reconPmmlObj.TransformationDictionary.DefineFunction[0].Apply.Extension[0].anytypeobjs_[0]
        content = base64.b64decode(content).decode()
        self.assertEqual(script_content, content)
        self.assertEqual(len(self.model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))

    @unittest.skip("")
    def test_04_plain_text_script(self):
        
        script_content = open("nyoka/tests/preprocess.py",'r').read()
        pmml_obj=KerasToPmml(self.model_final,
                    dataSet='image',
                    predictedClasses=['cat','dog'],
                    script_args = {
                        "content" : script_content,
                        "def_name" : "getBase64EncodedString",
                        "return_type" : "string",
                        "encode":False
                    }
                )
        pmml_obj.export(open("script_with_keras.pmml",'w'),0)
        self.assertEqual(os.path.isfile("script_with_keras.pmml"),True)
        reconPmmlObj = pml.parse("script_with_keras.pmml",True)
        content=reconPmmlObj.TransformationDictionary.DefineFunction[0].Apply.Extension[0].anytypeobjs_
        content[0] = content[0].replace("\t","")
        content="\n".join(content)
        self.assertEqual(script_content, content)
        self.assertEqual(len(self.model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))



    @classmethod
    def tearDownClass(cls):
        print("\n******* Finished *******\n")
     
if __name__ == '__main__':
    unittest.main(warnings='ignore')
        
        