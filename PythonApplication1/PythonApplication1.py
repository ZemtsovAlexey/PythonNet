from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import os

from data.ImageDataSet import ImageDataSet

net = ImageDataSet()

model = ''

if (os.path.exists('path_to_my_model.h5')):
    model = keras.models.load_model('path_to_my_model.h5')

if (not model):
    (dataset, imageCount) = net.load_dataset('C:/Users/zemtsov/Pictures/печать 3/')
    model = net.load_model()
    net.train(model, dataset, 3, imageCount)
    model.save('path_to_my_model.h5')

net.test('D:/documents types/доки/доки/upds', 'upds', model)

#image = net.get_iamge('C:/Users/zemtsov/Pictures/печать 2/test/false/statements_4624-0,9876425.png')
##image = net.load_iamge('https://static.beautyinsider.ru/2019/04/Vanya-do.jpg')
#predictions = model.predict(image)

#result = 'false'

#if (predictions[0][0] < predictions[0][1]):
#    result = 'true'

#print('\nТочность на проверочных данных:', result, predictions)