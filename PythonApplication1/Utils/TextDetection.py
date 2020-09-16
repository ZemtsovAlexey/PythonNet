import cv2
from PIL import Image


class TextDetection:

    @staticmethod
    def get_text_images(image_path):
        image = cv2.imread(image_path)

        w = 3000.
        height, width, depth = image.shape
        img_scale = w / width
        new_x, new_y = image.shape[1] * img_scale, image.shape[0] * img_scale
        image = cv2.resize(image, (int(new_x), int(new_y)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 30)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        contours = cv2.findContours(dilate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        roi_number = 0
        image_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area:
                x, y, w, h = cv2.boundingRect(contour)
                roi = image[y:y + h, x:x + w]
                pil_im = Image.fromarray(roi)
                image_list.append((pil_im, x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)
                # pil_im.show()
                # cv2.imwrite('ROI_{}.png'.format(roi_number), roi)
                roi_number += 1

        # cv2.imshow('thresh', thresh)
        # cv2.imshow('image', image)
        # cv2.waitKey()

        return image_list


td = TextDetection()
images = td.get_text_images('/mnt/c/rrr/Рисунок (392).jpg')
