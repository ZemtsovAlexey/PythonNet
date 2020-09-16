import cv2
import numpy as np
import math
from PIL import Image


class ImageUtils(object):

    @staticmethod
    def create_blank(width, height, rgb_color=(0, 0, 0)):
        image = np.zeros((height, width, 3), np.uint8)
        color = tuple(reversed(rgb_color))
        image[:] = color

        return image

    @staticmethod
    def resize_img(img, w):
        height, width, depth = img.shape
        img_scale = w / width
        new_x, new_y = img.shape[1] * img_scale, img.shape[0] * img_scale
        img = cv2.resize(img, (int(new_x), int(new_y)))

        return img

    @staticmethod
    def rotate_img(img, rt_degr):
        return img.rotate(rt_degr, expand=1)

    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    @staticmethod
    def orientation_correction(img, save_image = False):
        # GrayScale Conversion for the Canny Algorithm  
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # Canny Algorithm for edge detection was developed by John F. Canny not Kennedy!! :)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        # Using Houghlines to detect lines
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
        
        # Finding angle of lines in polar coordinates
        angles = []
        for x1, y1, x2, y2 in lines[0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # Getting the median angle
        median_angle = np.median(angles)
        
        # Rotating the image with this median angle
        # img_rotated = ndimage.rotate(img, median_angle)
        img_rotated = ImageUtils.rotate(img, median_angle)
        
        if save_image:
            cv2.imwrite('orientation_corrected.jpg', img_rotated)
        return img_rotated

    @staticmethod
    def OpenCvImgToPIL(image):
        im_pil = Image.fromarray(image)

        return im_pil
