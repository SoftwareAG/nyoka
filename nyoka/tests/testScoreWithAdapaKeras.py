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
    def setUpClass(self):
        print("******* Unit Test for Keras *******")
        self.adapa_utility = AdapaUtility()
        self.data_utility = DataUtility()


    def test_01_image_classifier_with_image_as_input(self):
        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
        activType='sigmoid'
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation=activType)(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
        
        cnn_pmml = KerasToPmml(model_final,model_name="MobileNetImage",description="Demo",\
            copyright="Internal User",dataSet='image',predictedClasses=['dogs','cats'])
        cnn_pmml.export(open('2classMBNet.pmml', "w"), 0)

        img = image.load_img('nyoka/tests/resizedCat.png')
        img = img_to_array(img)
        img = preprocess_input(img)
        imgtf = np.expand_dims(img, axis=0)
        model_pred=model_final.predict(imgtf)
        model_preds = {'dogs':model_pred[0][0],'cats':model_pred[0][1]}

        model_name  = self.adapa_utility.upload_to_zserver('2classMBNet.pmml')

        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, 'nyoka/tests/resizedCat.png',True)
  
        self.assertEqual(abs(probabilities['cats'] - model_preds['cats']) < 0.00001, True)
        self.assertEqual(abs(probabilities['dogs'] - model_preds['dogs']) < 0.00001, True)


    def test_02_image_classifier_with_base64string_as_input(self):
        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
        activType='sigmoid'
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation=activType)(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
        
        cnn_pmml = KerasToPmml(model_final,model_name="MobileNetBase64",description="Demo",\
            copyright="Internal User",dataSet='imageBase64',predictedClasses=['dogs','cats'])
        cnn_pmml.export(open('2classMBNetBase64.pmml', "w"), 0)

        img = image.load_img('nyoka/tests/resizedCat.png')
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

        predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, 'input.csv',True)
  
        self.assertEqual(abs(probabilities['cats'] - model_preds['cats']) < 0.00001, True)
        self.assertEqual(abs(probabilities['dogs'] - model_preds['dogs']) < 0.00001, True)


    # def test_03_SimpleDNN(self):
    #     X_train, X_test, y_train, columns, target_name, test_file = self.data_utility.get_data_for_regression()
    #     model = Sequential()
    #     model.add(Dense(13, input_dim=3, kernel_initializer='normal', activation='relu'))
    #     model.add(Dense(23))
    #     model.add(Dense(1, kernel_initializer='normal'))
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     model.fit(X_train, y_train, epochs=1000, verbose=0)
    #     pmmlObj=KerasToPmml(model, model_name='Sequential', description="Demo", copyright="Internal User",\
    #         dataSet="inputBase64")
    #     pmmlObj.export(open('sequentialModel.pmml','w'),0)

    #     base64string = "data:float32;base64," + FloatBase64.from_floatArray(np.array(X_test).flatten(),12)
    #     base64string = base64string.replace("\n", "")
    #     csvContent = "inputBase64\n\"" + base64string + "\""
    #     text_file = open(test_file, "w")
    #     text_file.write(csvContent)
    #     text_file.close()

    #     model_pred = model.predict(X_test)
    #     model_name  = self.adapa_utility.upload_to_zserver('sequentialModel.pmml')

    #     predictions, probabilities = self.adapa_utility.score_in_zserver(model_name, test_file,True)

    #     print(model_pred)
    #     print(predictions)
    #     print(probabilities)



    @classmethod
    def tearDownClass(self):
        print("\n******* Finished *******\n")
     
# if __name__ == '__main__':
#     unittest.main(warnings='ignore')
        
        