from __future__ import absolute_import, division, print_function, unicode_literals
from data.RotateModel import RotateModel

from tensorflow import keras
import os

net = RotateModel()

model = ''

#net.test2('D:/documents types/доки/testRotate')


if (os.path.exists('rotate_model.h5')):
    model = keras.models.load_model('rotate_model.h5')

if (not model):
    (dataset, imageCount) = net.load_dataset('D:/documents types/доки/testRotate')
    model = net.load_model()
    net.train(model, dataset, 3, imageCount)
    model.save('rotate_model.h5')

net.test('C:/Users/zemtsov/Pictures/печать 3/', model)