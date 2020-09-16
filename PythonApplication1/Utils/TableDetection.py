from collections import OrderedDict
import cv2
import math
import numpy as np
from tesserocr import PSM, OEM, RIL, PyTessBaseAPI, iterate_level
from ImageUtils import ImageUtils
from MathUtils import MathUtils, Point


class TableDetection(object):
    """TableDetection"""

    def __init__(self, src_img):
        self.__src_img = src_img
        self.__scale = 70

    def run(self):
        dif = 3

        hor_lines, ver_lines = self.__getLines()

        table = []
        rows = []

        for h in hor_lines:
            # cv2.line(img, (h.x1, h.y1), (h.x2, h.y2), (255, 0, 0), 5)
            # cv2.imshow("blank_image", img); cv2.waitKey()

            # ищем вертикальные линии пересекающиеся с текущей горозонтальной
            # intersecting_vertical_lines = self.__get_intersected_vertical_lines(ver_lines, h)
            intersecting_vertical_lines = [self.__intersection(h, line) for line in ver_lines]
            intersecting_vertical_lines = [line for line in intersecting_vertical_lines if line]

            # пропускаем горизонтальную линию если пересечений с вертикальными меньше двух
            if (len(intersecting_vertical_lines) < 2):
                continue

            # [cv2.line(img, (line[0].x1, line[0].y1), (line[0].x2, line[0].y2), (255, 0, 0), 5) for line in intersecting_vertical_lines]
            # cv2.imshow("blank_image", img); cv2.waitKey()

            last_row = rows[-1:][0] if len(rows[-1:]) > 0 else None

            # if last_row is not None and h.x2 - h.x1 < last_row.x2 - last_row.x1:
            #     continue

            rows += [self.__Line(h.x1, h.y1, h.x2, h.y2)]

            if len(rows) < 2:
                continue

            cells = []

            intersecting_vertical_lines = list(set(intersecting_vertical_lines))
            intersecting_vertical_lines.sort(key=lambda x: x[0].x1)

            # result = self.__get_intersected_vertical_lines([l[0] for l in intersecting_vertical_lines], last_row)
            result = [self.__intersection(last_row, line[0]) for line in intersecting_vertical_lines if line]
            result = [line for line in result if line]

            if len(result) == 0:
                continue

            prev_v = result[0]

            for v in result[1:]:
                # cells += [(prev_v.x1, last_row.y1, v.x1, h.y1)]
                cur_line_intersetc = self.__intersection(h, v[0])
                # pt1 = (int(prev_v[1][0]), int(prev_v[1][1]))
                # pt2 = (int(cur_line_intersetc[1][0]), int(cur_line_intersetc[1][1]))
                cells += [(int(prev_v[1][0]), int(prev_v[1][1]), int(cur_line_intersetc[1][0]), int(cur_line_intersetc[1][1]))]
                prev_v = v
                # i = cv2.rectangle(self.__src_img.copy(), pt1, pt2, (0, 0, 255), 3)
                # cv2.imshow('img', i)
                # cv2.waitKey(0)

            table += [cells]

        return ver_lines, hor_lines, table

    def __getLines(self):
        if len(self.__src_img.shape) == 2:
            gray_img = self.__src_img
        elif len(self.__src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.__src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        h_size = int(thresh_img.shape[1] / self.__scale)
        v_size = int(thresh_img.shape[0] / self.__scale)

        h_dilate_img = self.__getDilate(thresh_img, (30, 1))
        v_dilate_img = self.__getDilate(thresh_img, (1, 40))

        h_cords = self.__getHorizontalLines(h_dilate_img)
        v_cords = self.__getVerticalLines(v_dilate_img)

        # lines = h_cords + v_cords
        # [cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 2) for line in lines]
        # cv2.imshow("blank_image", img); cv2.waitKey()


        # kernel_size = 5

        # img_gray = cv2.cvtColor(self.__src_img, cv2.COLOR_BGR2GRAY) 
        # # blur_gray = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
        # img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        # cv2.imshow("blank_image", img_edges); cv2.waitKey()
        # # lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 10, minLineLength=50, maxLineGap=3)
        # lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 50, np.array([]), 100, 3)

        # [cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2) for line in lines]
        # cv2.imshow("blank_image", img); cv2.waitKey()


        hor_lines = h_cords
        ver_lines = v_cords

        # hor_lines = []
        # ver_lines = []

        # for h in h_cords:
        #     res = self.__get_intersected_vertical_lines(v_cords, h)

        #     if len(res) > 1:
        #         ver_lines += res

        # for v in v_cords:
        #     res = self.__get_intersected_horizontal_lines(h_cords, v)

        #     if len(res) > 1:
        #         hor_lines += res

        hor_lines = list(set(hor_lines))
        hor_lines.sort(key=lambda x: x.y1 and x.y2)

        ver_lines = list(set(ver_lines))
        ver_lines.sort(key=lambda x: x.x1 and x.x2)

        # lines = hor_lines + ver_lines
        # [cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (0, 0, 255), 2) for line in lines]
        # cv2.imshow("blank_image", img); cv2.waitKey()

        return hor_lines, ver_lines

    def __get_intersected_vertical_lines(self, lines: list, target_line):
        dif = 10

        result = [line for line in lines if
                  ((target_line.x1 - dif <= line.x1 <= target_line.x2 + dif) or (
                              target_line.x1 - dif <= line.x2 <= target_line.x2 + dif)) and
                  ((line.y1 - dif <= target_line.y1 <= line.y2 + dif) or (
                              line.y1 - dif <= target_line.y2 <= line.y2 + dif))]

        # result = [self.__intersection(target_line, line) for line in lines if line]

        return result

    @staticmethod
    def __get_intersected_horizontal_lines(lines: list, target_line):
        dif = 10

        result = [line for line in lines if
                  ((target_line.y1 - dif <= line.y1 <= target_line.y2 + dif) or (
                              target_line.y1 - dif <= line.y2 <= target_line.y2 + dif)) and
                  ((line.x1 - dif <= target_line.x1 <= line.x2 + dif) or (
                              line.x1 - dif <= target_line.x2 <= line.x2 + dif))]

        return result

    @staticmethod
    def __get_line_intersect(line_a, line_b):
        def has_point(x, y):
            dif = 3
            return min(line_a.x1, line_a.x2) - dif <= x <= max(line_a.x1, line_a.x2) + dif and \
                   min(line_a.y1, line_a.y2) - dif <= y <= max(line_a.y1, line_a.y2) + dif

        A1 = line_a.y1 - line_a.y2
        B1 = line_a.x2 - line_a.x1
        C1 = line_a.x1 * line_a.y2 - line_a.x2 * line_a.y1

        A2 = line_b.y1 - line_b.y2
        B2 = line_b.x2 - line_b.x1
        C2 = line_b.x1 + line_b.y2 - line_b.x2 * line_b.y1

        if B1 * A2 - B2 * A1 and A1:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C1 - B1 * y) / A1
            return (x, y) if has_point(x, y) else None

        elif B1 * A2 - B2 * A1 and A2:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C2 - B2 * y) / A2
            return (x, y) if has_point(x, y) else None

        else:
            return None

    @staticmethod
    def __intersection(L1, L2):
        dif = 5

        # if ((min(L2.x1, L2.x2) - dif <= (L1.x1 or L1.x2) <= max(L2.x1, L2.x2) + dif) and
        #     (min(L2.y1, L2.y2) - dif <= (L1.y1 or L1.y2) <= max(L2.y1, L2.y2) + dif)):
        #     return False

        # if (((min(L2.x1, L2.x2) - dif <= L1.x1 <= max(L2.x1, L2.x2) + dif) or (min(L2.x1, L2.x2) - dif <= L1.x2 <= max(L2.x1, L2.x2) + dif)) and 
        #     ((min(L2.y1, L2.y2) - dif <= L1.y1 <= max(L2.y1, L2.y2) + dif) or (min(L2.y1, L2.y2) - dif <= L1.y2 <= max(L2.y1, L2.y2) + dif))):
        #     return False

        try:
            D = float(L1.A) * float(L2.B) - float(L1.B) * float(L2.A)
            Dx = float(L1.C) * float(L2.B) - float(L1.B) * float(L2.C)
            Dy = float(L1.A) * float(L2.C) - float(L1.C) * float(L2.A)

            if D != 0:
                x = Dx / D
                y = Dy / D

                if (min(L1.x1, L1.x2) - dif <= x <= max(L1.x1, L1.x2) + dif and 
                    min(L2.y1, L2.y2) - dif <= y <= max(L2.y1, L2.y2) + dif):
                    return L2, (x, y)
                
                return False
            else:
                return False
        except OverflowError:
            return False

    @staticmethod
    def __getDilate(thresh_img, k_size: int):
        image = thresh_img.copy()
        structure = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
        erode_img = cv2.erode(image, structure, 1)
        dilate_img = cv2.dilate(erode_img, structure, 1)

        return dilate_img

    def __getVerticalLines(self, dilate_img):
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cords = [(min(contour, key=lambda v: v[0][1])[0], max(contour, key=lambda v: v[0][1])[0]) for contour in
                 contours]
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

            self.A = self.y1 - self.y2
            self.B = self.x2 - self.x1
            self.C = -(self.x1 * self.y2 - self.x2 * self.y1)

        x1 = 0
        y2 = 0
        x2 = 0
        y2 = 0

        A = 0
        B = 0
        C = 0


# fileName = '/mnt/c/rrr/Рисунок (410).jpg'
fileName = '/mnt/c/rrr/Рисунок (392).jpg'

img = cv2.imread(fileName)
# img = ImageUtils.resize_img(img, 2500.)
img = ImageUtils.orientation_correction(img, False)
ver_lines, hor_lines, table = TableDetection(img).run()

height, width, depth = img.shape
blank_image = ImageUtils.create_blank(width, height, (255, 255, 255))
img = img.copy()

# for rows in table:
#     for row in rows:
#         x1, y1, w, h = row
#         img = cv2.rectangle(img, (int(x1), int(y1)), (w, h), (0, 255, 0), 5)
        # img = cv2.polylines(img, np.array([[25, 70], [25, 145]], np.int32), True, (0, 255, 0), 5)

# cv2.imshow("blank_image", img)
# cv2.waitKey()

im_pil = ImageUtils.OpenCvImgToPIL(img)
padding = -1

# for rows in table:
#     for row in rows:
#         x, y, w, h = row

#         with PyTessBaseAPI(lang='rus+eng') as api:
            
#             # crop_rectangle = (x, y, w, h)
#             # cropped_im = im_pil.crop(crop_rectangle)
#             # cropped_im = ImageUtils.resize_img(cropped_im, 200.)

#             crop_img = img[y:h, x:w]
#             crop_img = ImageUtils.resize_img(crop_img, 500.)
#             cropped_im = ImageUtils.OpenCvImgToPIL(crop_img)

#             api.SetImage(cropped_im)
#             cropped_im.show()
#             cv2.waitKey(0)
#             symbol = api.GetUTF8Text()
#             print(symbol)
#             print('---------------------------------------------')

with PyTessBaseAPI(lang='rus+eng', oem=OEM.DEFAULT) as api:
    api.SetImage(im_pil)
    api.SetPageSegMode(PSM.SINGLE_COLUMN)

    for rows in table:
        for row in rows:
            x, y, w, h = row

            # x += padding
            # y += padding
            # w -= padding * 2 + 1
            # h -= padding * 2 + 1

            api.SetRectangle(x, y, w - x, h - y)
            api.SetPageSegMode(PSM.SINGLE_BLOCK)
            api.Recognize()

            level = RIL.BLOCK
            r = api.GetIterator()
            conf = r.Confidence(level)
            # print(conf)

            if conf < 60:
                api.SetPageSegMode(PSM.SINGLE_LINE)
                api.Recognize()

            symbol = api.GetUTF8Text()
            print(symbol)
            print('---------------------------------------------')

            


            # ri = api.GetIterator()

            # if ri is None:
            #     continue

            # level = RIL.PARA

            # for r in iterate_level(ri, level):
            #     if r:
            #         try:
            #             symbol = r.GetUTF8Text(level)  # r == ri
            #             conf = r.Confidence(level)
            #             if symbol:
            #                 print(symbol)
            #             indent = False
            #             ci = r.GetChoiceIterator()
            #             for c in ci:
            #                 choice = c.GetUTF8Text()  # c == ci
            #                 indent = True
            #         except Exception:
            #             continue
            
            # print('---------------------------------------------')

            i = cv2.rectangle(img.copy(), (x, y), (w, h), (0, 0, 255), 3)
            cv2.imshow('img', i)
            cv2.waitKey(0)