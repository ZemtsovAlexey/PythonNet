from __future__ import absolute_import, division, print_function, unicode_literals
from data.TypingModel import TypingModel

from tensorflow import keras
import os

net = TypingModel()

model = ''

if (os.path.exists('typing_model.h5')):
    model = keras.models.load_model('typing_model.h5')

if (not model):
    (dataset, imageCount) = net.load_dataset('D:/documents types/доки/teach')
    model = net.load_model()
    net.train(model, dataset, 3, imageCount)
    model.save('typing_model.h5')

net.test('C:/Users/zemtsov/Pictures/печать 3/', model)