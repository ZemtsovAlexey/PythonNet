import json
import os
import cv2
import numpy as np
import random

from tesserocr import PyTessBaseAPI, RIL, iterate_level, PSM
import tesserocr
from PIL import Image
from PythonApplication1.Utils.ImageUtils import ImageUtils


class CutImage(object):
    def __init__(self, img, bin_threshold, kernel, iterations, areaRange, filename, border=10, show=True, write=True, ):
        self.img = img
        self.bin_threshold = bin_threshold
        self.kernel = kernel
        self.iterations = iterations
        self.areaRange = areaRange
        self.border = border
        self.show = show
        self.write = write
        self.filename = filename

    def GetResult(self):
        fl = open(self.filename, 'w')

        if self.img.shape[2] == 1:
            img_gray = self.img
        elif self.img.shape[2] == 3:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img_gray, self.bin_threshold, 255, cv2.THRESH_BINARY_INV)  # 二值化
        img_erode = cv2.dilate(thresh, self.kernel, iterations=self.iterations)

        # cv2.imshow('thresh', thresh)
        # cv2.imshow('erode', img_erode)
        # cv2.waitKey()

        # image, contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        roiList = []
        res = []
        result = {}
        area_coord_roi = []

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            if self.areaRange[0] < area < self.areaRange[1]:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = self.img[y + self.border:(y + h) - self.border, x + self.border:(x + w) - self.border]
                area_coord_roi.append((area, (x, y, w, h), roi))

        max_area = max([info[0] for info in area_coord_roi])

        for info in area_coord_roi:
            if info[0] == max_area:
                max_rect = info[1]

        for each in area_coord_roi:
            x, y, w, h = each[1]
            if x > max_rect[0] and y > max_rect[1] and (x + w) < (max_rect[0] + max_rect[2]) and (y + h) < (
                    max_rect[1] + max_rect[3]):
                pass
            else:
                tmp_ = each[1]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # tmp = []
                # name = "tmp.jpg"
                # cv2.imwrite(name, each[2])
                # tmp.append(" ")
                # tmp.extend(list(tmp_))
                # tmp.append("0 0 0")
                # res.append(tmp)
                # os.remove(name)

        cv2.imshow("yyy", img)

        result['1'] = [res]
        # fl.write(json.dumps(result))

        return roiList


class DetectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
        elif len(self.src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 70

        h_size = int(h_img.shape[1] / scale)

        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        h_erode_img = cv2.erode(h_img, h_structure, 1)
        h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)

        h_contours, h_hierarchy = cv2.findContours(h_dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

        mask_img = h_dilate_img + v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)

        return mask_img, joints_img, h_contours, h_dilate_img, v_dilate_img


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def resize_img(img, w):
    height, width, depth = img.shape
    img_scale = w / width
    new_x, new_y = img.shape[1] * img_scale, img.shape[0] * img_scale
    img = cv2.resize(img, (int(new_x), int(new_y)))

    return img


if __name__ == '__main__':

    # fileName = r'C:\trash\docs\4.jpg'
    # fileName = r'C:\trash\docs\invoices\115653180\0.jpg'
    fileName = r'C:\trash\docs\statements\119743359\3.jpg'
    # fileName = r'C:\trash\docs\statements\121535680\0.jpg'
    # fileName = r'C:\trash\docs\tovarnaya-nakladnaya-torg-12-obrazec.jpg'

    img = cv2.imread(fileName)
    img = resize_img(img, 3000.)

    mask, joint, joints_contours, h_dilate_img, v_dilate_img = DetectTable(img).run()

    # cv2.imshow("h_structure", joint)
    # cv2.waitKey()

    table_image_contour = mask + joint
    table_image = img
    ret, thresh_value = cv2.threshold(table_image_contour, 180, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(table_image_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h_contours, h_hierarchy = cv2.findContours(h_dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_contours, v_hierarchy = cv2.findContours(v_dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cv2.boundingRect(cnt) for cnt in contours]
    v_cord = [(min(cnt, key=lambda v: v[0][1])[0], max(cnt, key=lambda v: v[0][1])[0]) for cnt in v_contours]
    h_cord = [(min(cnt, key=lambda v: v[0][0])[0], max(cnt, key=lambda v: v[0][0])[0]) for cnt in h_contours]

    hor_lines = list()
    ver_lines = list()
    dif = 10

    for cnt in h_cord:
        l, r = cnt
        res = [item for item in v_cord if (l[0] - dif <= item[0][0] <= r[0] + dif) and (l[1] - dif <= item[0][1] <= r[1] + dif)]

        if len(res) > 0:
            ver_lines += res

    for cnt in v_cord:
        l, r = cnt
        res = [item for item in h_cord if (l[0] - dif <= item[0][0] <= r[0] + dif) and (l[1] - dif <= item[0][1] <= r[1] + dif)]

        if len(res) > 0:
            hor_lines += res

    def sortY(val):
        return val[0][1]

    def sortX(val):
        return val[0][0]

    hor_lines.sort(key=sortY)
    ver_lines.sort(key=sortX)

    rows = len(hor_lines)




    height, width, depth = table_image.shape
    blank_image = create_blank(width, height, (255, 255, 255))

    for cnt in ver_lines + hor_lines:
        l, r = cnt
        img = cv2.line(img, (l[0], l[1]), (r[0], r[1]), (0, 0, 255), 5)

    # cv2.imshow("blank_image", img)
    # cv2.waitKey()

    gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image = create_blank(width, height, (255, 255, 255))
    im_pil = Image.fromarray(img)
    padding = 0

    with PyTessBaseAPI(lang='rus+eng') as api:
            api.SetImage(im_pil)
            api.Recognize()

            page = api.GetIterator()
            o = page.Orientation()
            print(o)
            print(api.DetectOrientationScript())

            # im_pil.show()
            img = ImageUtils.rotate_img(im_pil, o[3])
            img.show()



            # for cnt in contours:
            #     area = cv2.contourArea(cnt)
            #     if area > 500:
            #         x, y, w, h = cv2.boundingRect(cnt)
            #         x += padding
            #         y += padding
            #         w -= padding * 2 + 1
            #         h -= padding * 2 + 1
            #
            #         # api.SetVariable("save_blob_choices", "T")
            #         # api.SetVariable("textord_tabfind_find_tables", "true")
            #         # api.SetVariable("textord_show_tables", "true")
            #         # api.SetVariable("textord_tablefind_recognize_tables", "true")
            #         api.SetRectangle(x, y, w, h)
            #         api.SetPageSegMode(PSM.SINGLE_BLOCK)
            #         # api.SetPageSegMode(PSM.COUNT)
            #         # print(api.GetUTF8Text())
            #         api.Recognize()
            #
            #         level = RIL.BLOCK
            #         r = api.GetIterator()
            #         conf = r.Confidence(level)
            #         print(conf)
            #
            #         if conf < 50:
            #             api.SetPageSegMode(PSM.SINGLE_LINE)
            #
            #         api.Recognize()
            #         symbol = api.GetUTF8Text()
            #         # symbol = r.GetUTF8Text(level)
            #         print(symbol)
            #         print('---------------------------------------------')
            #
            #         i = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 3)
            #         cv2.imshow('img', i)
            #         cv2.waitKey(0)
