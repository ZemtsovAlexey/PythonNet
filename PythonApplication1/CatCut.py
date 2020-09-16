from __future__ import print_function
from PIL import Image
import cv2
import numpy as np

image_path = r'C:\trash\scale_1200.jpg'
image = cv2.imread(image_path)

w = 2000.
height, width, depth = image.shape
img_scale = w / width
new_x, new_y = image.shape[1] * img_scale, image.shape[0] * img_scale
image = cv2.resize(image, (int(new_x), int(new_y)))

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 10, 20)
ret, thresh = cv2.threshold(blur, 65, 255, cv2.THRESH_BINARY)

# im_pil = Image.fromarray(image)

# cv2.imwrite(r'C:\trash\cat.jpg', thresh)
cv2.imshow('cat', thresh)
cv2.waitKey(0)
