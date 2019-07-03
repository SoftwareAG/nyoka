import os
import unittest
from keras import applications
from keras.layers import *
from keras.models import Model
from nyoka import KerasToPmml
from nyoka import PMML43Ext as ny
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential


class TestMethods(unittest.TestCase):

    
    def test_keras_01(self):

        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3))
        activType='sigmoid'
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation=activType)(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')
        cnn_pmml = KerasToPmml(model_final,dataSet='image',predictedClasses=['cats','dogs'])
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


if __name__=='__main__':
    unittest.main(warnings='ignore')







