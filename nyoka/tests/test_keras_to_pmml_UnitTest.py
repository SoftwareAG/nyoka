
import sys,os


import unittest
from keras import applications
from keras.layers import *
from keras.models import Model
from nyoka import KerasToPmml
from nyoka import PMML44 as ny
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
import base64


class TestMethods(unittest.TestCase):

    
    def test_keras_01(self):

        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
        activType='sigmoid'
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation=activType)(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
        cnn_pmml = KerasToPmml(model_final,model_name="MobileNet",description="Demo",\
            copyright="Internal User",dataSet='image',predictedClasses=['cats','dogs'])
        cnn_pmml.export(open('2classMBNet.pmml', "w"), 0)
        reconPmmlObj=ny.parse('2classMBNet.pmml',True)
        self.assertEqual(os.path.isfile("2classMBNet.pmml"),True)
        self.assertEqual(len(model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))


    def test_keras_02(self):
        boston = load_boston()
        data = pd.DataFrame(boston.data)
        features = list(boston.feature_names)
        target = 'PRICE'
        data.columns = features
        data['PRICE'] = boston.target
        x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.20, random_state=42)
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(23))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1000, verbose=0)
        pmmlObj=KerasToPmml(model)
        pmmlObj.export(open('sequentialModel.pmml','w'),0)
        reconPmmlObj=ny.parse('sequentialModel.pmml',True)
        self.assertEqual(os.path.isfile("sequentialModel.pmml"),True)
        self.assertEqual(len(model.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer)-1)

    def test_encoded_script(self):

        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation='sigmoid')(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
        script_content = open("nyoka/tests/preprocess.py",'r').read()
        pmml_obj=KerasToPmml(model_final,
                    dataSet='image',
                    predictedClasses=['cat','dog'],
                    script_args = {
                        "content" : script_content,
                        "def_name" : "getBase64EncodedString",
                        "return_type" : "string",
                        "encode":True
                    }
                )
        pmml_obj.export(open("script_with_keras_encoded.pmml",'w'),0)
        self.assertEqual(os.path.isfile("script_with_keras_encoded.pmml"),True)
        reconPmmlObj = ny.parse("script_with_keras_encoded.pmml",True)
        content=reconPmmlObj.TransformationDictionary.DefineFunction[0].Apply.Extension[0].anytypeobjs_[0]
        content = base64.b64decode(content).decode()
        self.assertEqual(script_content, content)
        self.assertEqual(len(model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))

    def test_plain_text_script(self):

        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation='sigmoid')(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
        script_content = open("nyoka/tests/preprocess.py",'r').read()
        pmml_obj=KerasToPmml(model_final,
                    dataSet='image',
                    predictedClasses=['cat','dog'],
                    script_args = {
                        "content" : script_content,
                        "def_name" : "getBase64EncodedString",
                        "return_type" : "string",
                        "encode":False
                    }
                )
        pmml_obj.export(open("script_with_keras_plain.pmml",'w'),0)
        self.assertEqual(os.path.isfile("script_with_keras_plain.pmml"),True)
        reconPmmlObj = ny.parse("script_with_keras_plain.pmml",True)
        content=reconPmmlObj.TransformationDictionary.DefineFunction[0].Apply.Extension[0].anytypeobjs_
        content[0] = content[0].replace("\t","")
        content="\n".join(content)
        self.assertEqual(script_content, content)
        self.assertEqual(len(model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))


if __name__=='__main__':
    unittest.main(warnings='ignore')







