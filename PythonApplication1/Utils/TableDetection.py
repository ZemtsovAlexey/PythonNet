import json
import os
import cv2
import numpy as np
import random
from collections import OrderedDict
from PythonApplication1.Utils.ImageUtils import ImageUtils
from PythonApplication1.Utils.MathUtils import MathUtils, Point
from tesserocr import PyTessBaseAPI, RIL, iterate_level, PSM


class TableDetection(object):
    def __init__(self, src_img):
        self.__src_img = src_img
        self.__scale = 50

    def run(self):
        if len(self.__src_img.shape) == 2:
            gray_img = self.__src_img
        elif len(self.__src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.__src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        h_size = int(thresh_img.shape[1] / self.__scale)
        v_size = int(thresh_img.shape[0] / self.__scale)

        h_dilate_img = self.__getDilate(thresh_img, (h_size, 1))
        v_dilate_img = self.__getDilate(thresh_img, (1, v_size))

        h_cords = self.__getHorizontalLines(h_dilate_img)
        v_cords = self.__getVerticalLines(v_dilate_img)

        hor_lines = []
        ver_lines = []
        dif = 3

        for h in h_cords:
            res = [v for v in v_cords if
                   ((h.x1 - dif <= v.y1 <= h.x2 + dif) or (h.x1 - dif <= v.x2 <= h.x2 + dif)) and
                   ((v.y1 - dif <= h.y1 <= v.y2 + dif) or (v.y1 - dif <= h.y2 <= v.y2 + dif))]

            if len(res) > 1:
                ver_lines += res

        for v in v_cords:
            res = [h for h in h_cords if
                   ((v.y1 - dif <= h.y1 <= v.y2 + dif) or (v.y1 - dif <= h.y2 <= v.y2 + dif)) and
                   ((h.x1 - dif <= v.x1 <= h.x2 + dif) or (h.x1 - dif <= v.x2 <= h.x2 + dif))]

            if len(res) > 1:
                hor_lines += res

        hor_lines = list(set(hor_lines))
        hor_lines.sort(key=lambda x: x.y1 and x.y2)

        ver_lines = list(set(ver_lines))
        ver_lines.sort(key=lambda x: x.x1 and x.x2)

        table = []
        rows = []

        for h in hor_lines:
            res = [v for v in v_cords if
                   ((h.x1 - dif <= v.x1 <= h.x2 + dif) or (h.x1 - dif <= v.x2 <= h.x2 + dif)) and
                   ((v.y1 - dif <= h.y1 <= v.y2 + dif) or (v.y1 - dif <= h.y2 <= v.y2 + dif))]

            if (len(res) < 2):
                continue

            last_row = rows[-1:][0] if len(rows[-1:]) > 0 else None

            # if last_row is not None and h.x2 - h.x1 < last_row.x2 - last_row.x1:
            #     continue

            rows += [self.__Line(h.x1, h.y1, h.x2, h.y2)]

            if len(rows) < 2:
                continue

            cells = []

            res = list(set(res))
            res.sort(key=lambda x: x.x1)

            res = [v for v in res if
                   ((last_row.x1 - dif <= v.x1 <= last_row.x2 + dif) or (last_row.x1 - dif <= v.x2 <= last_row.x2 + dif)) and
                   ((v.y1 - dif <= last_row.y1 <= v.y2 + dif) or (v.y1 - dif <= last_row.y2 <= v.y2 + dif))]

            if len(res) == 0:
                continue

            prev_v = res[0]

            for v in res[1:]:
                cells += [(prev_v.x1, last_row.y1, v.x1, h.y1)]
                prev_v = v

            table += [cells]

        return ver_lines, hor_lines, table

    def rotateByMaxHovLine(self):
        if len(self.__src_img.shape) == 2:
            gray_img = self.__src_img
        elif len(self.__src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.__src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        h_size = int(thresh_img.shape[1] / self.__scale)
        h_dilate_img = self.__getDilate(thresh_img, (h_size, 1))
        h_cords = self.__getHorizontalLines(h_dilate_img)

        max_len = self.__Line(0, 0, 0, 0)

        for i in h_cords:
            if i.x2 - i.x1 > max_len.x2 - max_len.x1:
                max_len = i

        p1 = Point(max_len.x1, max_len.y1)
        p2 = Point(max_len.x2, max_len.y2)

        angel = MathUtils.GetAngleOfLineBetweenTwoPoints(p1, p2)
        self.__src_img = ImageUtils.rotate(self.__src_img, angel)

        return angel

    @staticmethod
    def __getDilate(thresh_img, k_size):
        image = thresh_img.copy()
        structure = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
        erode_img = cv2.erode(image, structure, 1)
        dilate_img = cv2.dilate(erode_img, structure, 1)

        return dilate_img

    def __getVerticalLines(self, dilate_img):
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cords = [(min(contour, key=lambda v: v[0][1])[0], max(contour, key=lambda v: v[0][1])[0]) for contour in
                 contours]
        # cords = [(i[0][0], i[0][1], i[1][0], i[1][1]) for i in cords]
        cords = [self.__Line(i[0][0], i[0][1], i[1][0], i[1][1]) for i in cords]

        return cords

    def __getHorizontalLines(self, dilate_img):
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cords = [(min(contour, key=lambda v: v[0][0])[0], max(contour, key=lambda v: v[0][0])[0]) for contour in
                 contours]
        cords = [self.__Line(i[0][0], i[0][1], i[1][0], i[1][1]) for i in cords]

        return cords

    class __Line:
        def __init__(self, x1, y1, x2, y2):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2

        x1 = 0
        y2 = 0
        x2 = 0
        y2 = 0


# fileName = r'C:\trash\docs\4.jpg'
fileName = r'C:\trash\docs\invoices\115653180\0.jpg'
# fileName = r'C:\trash\docs\statements\121535680\0.jpg'
# fileName = r'C:\trash\docs\statements\119743359\3.jpg'
# fileName = r'C:\trash\docs\statements\156046082\4.jpg'
# fileName = r'C:\trash\docs\statements\156734950\0.jpg'
# fileName = r'C:\trash\docs\tovarnaya-nakladnaya-torg-12-obrazec.jpg'

img = cv2.imread(fileName)
img = ImageUtils.resize_img(img, 2000.)

tableDetection = TableDetection(img)
angel = tableDetection.rotateByMaxHovLine()
img = ImageUtils.rotate(img, angel)
ver_lines, hor_lines, table = TableDetection(img).run()

height, width, depth = img.shape
blank_image = ImageUtils.create_blank(width, height, (255, 255, 255))
img = img.copy()

for rows in table:
    for row in rows:
        x1, y1, w, h = row
        img = cv2.rectangle(img, (x1, y1), (w, h), (0, 255, 0), 5)

cv2.imshow("blank_image", img)
cv2.waitKey()

im_pil = ImageUtils.OpenCvImgToPIL(img)
padding = 1

# with PyTessBaseAPI(lang='rus+eng') as api:
#     api.SetImage(im_pil)
#     for rows in table:
#         for row in rows:
#             x, y, w, h = row
#
#             x += padding
#             y += padding
#             w -= padding * 2 + 1
#             h -= padding * 2 + 1
#
#             api.SetRectangle(x, y, w - x, h - y)
#             api.SetPageSegMode(PSM.SINGLE_BLOCK)
#             api.Recognize()
#
#             level = RIL.BLOCK
#             r = api.GetIterator()
#             conf = r.Confidence(level)
#             print(conf)
#
#             if conf < 80:
#                 api.SetPageSegMode(PSM.SINGLE_LINE)
#
#             api.Recognize()
#             symbol = api.GetUTF8Text()
#             print(symbol)
#             print('---------------------------------------------')
#
#             i = cv2.rectangle(img.copy(), (x, y), (w, h), (0, 0, 255), 3)
#             cv2.imshow('img', i)
#             cv2.waitKey(0)
