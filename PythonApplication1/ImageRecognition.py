from __future__ import print_function
from tesserocr import PyTessBaseAPI, RIL, iterate_level
from PIL import Image
import tesserocr
import cv2
import numpy as np

image_path = r'C:\trash\docs\4.jpg'
image = cv2.imread(image_path)

w = 2000.
height, width, depth = image.shape
img_scale = w / width
new_x, new_y = image.shape[1] * img_scale, image.shape[0] * img_scale
image = cv2.resize(image, (int(new_x), int(new_y)))

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)
im_pil = Image.fromarray(image)



with PyTessBaseAPI(lang='rus+eng') as api:
    api.SetImage(im_pil)
    api.SetVariable("save_blob_choices", "T")
    api.SetVariable("textord_tabfind_find_tables", "true")
    # api.SetVariable("textord_show_tables", "true")
    api.SetVariable("textord_tablefind_recognize_tables", "true")
    # api.SetRectangle(37, 228, 548, 31)
    # print(api.GetUTF8Text())
    api.Recognize()

    ri = api.GetIterator()
    level = RIL.BLOCK

    for r in iterate_level(ri, level):
        if ri.BlockType() == tesserocr.PT.TABLE:
        print("found a table\n")
        x1, y1, x2, y2 = ri.BoundingBox(level)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        symbol = r.GetUTF8Text(level)
        print(symbol)
        print("table BoundingBox: ", x1, y1, x2, y2)
        print('---------------------------------------------')

        # symbol = r.GetUTF8Text(level)  # r == ri
        conf = r.Confidence(level)
        # if symbol:
        #     print(u'symbol {}, conf: {}'.format(symbol, conf), end='')
        # indent = False
        # ci = r.GetChoiceIterator()
        # for c in ci:
        #     if indent:
        #         print('\t\t ', end='')
        #     print('\t- ', end='')
        #     choice = c.GetUTF8Text()  # c == ci
        #     print(u'{} conf: {}'.format(choice, c.Confidence()))
        #     indent = True
        # print('---------------------------------------------')

    cv2.imshow('img', image)
    cv2.waitKey(0)
