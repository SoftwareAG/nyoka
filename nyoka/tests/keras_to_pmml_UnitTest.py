import os
import unittest
from keras import applications
from keras.layers import Flatten, Dense
from keras.models import Model
from nyoka import KerasToPmml


class TestMethods(unittest.TestCase):

    
    def test_keras_01(self):

        model = applications.MobileNet(weights='imagenet', include_top=False,input_shape = (224, 224,3)) #last layer not included

        activType='sigmoid'
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation=activType)(x)
        model_final = Model(inputs =model.input, outputs = predictions,name='predictions')

        cnn_pmml = KerasToPmml(model_final,predictedClasses=['cats','dogs'])

        cnn_pmml.export(open('2classMBNet.pmml', "w"), 0)

        self.assertEqual(os.path.isfile("2classMBNet.pmml"),True)


if __name__=='__main__':
    unittest.main(warnings='ignore')







