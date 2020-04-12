import cv2
import numpy as np
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
    def OpenCvImgToPIL(image):
        im_pil = Image.fromarray(image)

        return im_pil
