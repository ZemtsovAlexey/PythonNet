from __future__ import absolute_import, division, print_function, unicode_literals
from models.TextModel import TextModel
from Utils.TextDetection import TextDetection

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tensorflow import keras
import os
import cv2
import numpy as np

net = TextModel()
detection = TextDetection()

model = ''
model_name = 'text_model2.h5'

if os.path.exists(model_name):
    model = keras.models.load_model(model_name)

if not model:
    (data_set, imageCount) = net.load_dataset('PythonApplication1/images/fonts')
    model = net.load_model()
    net.train(model, data_set, 5, imageCount)
    model.save(model_name)

image_path = '/mnt/c/rrr/Рисунок (392).jpg'
images = detection.get_text_images(image_path)
image = Image.open(image_path)

base_width = 2000
w_percent = (base_width / float(image.size[0]))
h_size = int((float(image.size[1]) * float(w_percent)))
image = image.resize((base_width, h_size), Image.ANTIALIAS)

result_image = Image.new('L', (image.width, image.height), 215)
draw = ImageDraw.Draw(result_image)
font = ImageFont.truetype('PythonApplication1/fonts/arial.ttf', 12)

batch_holder = []

for i, im in enumerate(images):
    batch_holder.append(im[0])

predictions = net.predict(model, batch_holder)

for i, im in enumerate(images):
    img,x,y,w,h = im
    character = predictions[i]
    draw.text((x, y), character, (245-215), font=font)

# result_image.show()
opencvImage = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
cv2.imshow('thresh', opencvImage)
cv2.waitKey()