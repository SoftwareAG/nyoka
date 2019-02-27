import unittest
from keras import models
from keras import layers,optimizers
from keras import applications
import keras
import os
from nyoka.keras.keras_model_to_pmml import KerasToPmml
from nyoka import PMML43Ext as ny

class TestMethods(unittest.TestCase):

    def test_construction_mobilenet(self):
        model = applications.MobileNet(weights = "imagenet", include_top=False,input_shape = (224, 224, 3))
        x = model.output
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation="relu")(x)
        predictions = layers.Dense(2, activation="softmax")(x)
        model_final = models.Model(input = model.input, output = predictions)
        model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
        pmmlObj=KerasToPmml(model_final,dataSet='image')
        pmmlObj.export(open('mobilenet.pmml','w'),0)
        reconPmmlObj=ny.parse('mobilenet.pmml',True)
        self.assertEqual(os.path.isfile("mobilenet.pmml"),True)
        self.assertEqual(len(model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))

    def test_construction_inception(self):
        model = applications.InceptionV3(weights = "imagenet", include_top=False,input_shape = (224, 224, 3))
        x = model.output
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation="relu")(x)
        predictions = layers.Dense(2, activation="softmax")(x)
        model_final = models.Model(input = model.input, output = predictions)
        model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
        pmmlObj=KerasToPmml(model_final,dataSet='image')
        pmmlObj.export(open('inception.pmml','w'),0)
        reconPmmlObj=ny.parse('inception.pmml',True)
        self.assertEqual(os.path.isfile("inception.pmml"),True)
        self.assertEqual(len(model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))

    def test_construction_resnet(self):
        model = applications.ResNet50(weights = "imagenet", include_top=False,input_shape = (224, 224, 3))
        x = model.output
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation="relu")(x)
        predictions = layers.Dense(2, activation="softmax")(x)
        model_final = models.Model(input = model.input, output = predictions)
        model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
        pmmlObj=KerasToPmml(model_final,dataSet='image')
        pmmlObj.export(open('resnet.pmml','w'),0)
        reconPmmlObj=ny.parse('resnet.pmml',True)
        self.assertEqual(os.path.isfile("resnet.pmml"),True)
        self.assertEqual(len(model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))

    def test_construction_vgg(self):
        model = applications.VGG16(weights = "imagenet", include_top=False,input_shape = (224, 224, 3))
        x = model.output
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation="relu")(x)
        predictions = layers.Dense(2, activation="softmax")(x)
        model_final = models.Model(input = model.input, output = predictions)
        model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
        pmmlObj=KerasToPmml(model_final,dataSet='image')
        pmmlObj.export(open('vgg.pmml','w'),0)
        reconPmmlObj=ny.parse('vgg.pmml',True)
        self.assertEqual(os.path.isfile("vgg.pmml"),True)
        self.assertEqual(len(model_final.layers), len(reconPmmlObj.DeepNetwork[0].NetworkLayer))


if __name__=='__main__':
    unittest.main(warnings='ignore')