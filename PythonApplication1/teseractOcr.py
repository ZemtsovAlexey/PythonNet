import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
image_path = r'C:\trash\docs\statements\151051412\0.jpg'
# img = Image.open(image_path)
image = cv2.imread(image_path)

w = 1500.
height, width, depth = image.shape
img_scale = w / width
new_x, new_y = image.shape[1] * img_scale, image.shape[0] * img_scale
image = cv2.resize(image, (int(new_x), int(new_y)))

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)
im_pil = Image.fromarray(thresh)
# cv2.imshow('ddd', thresh)

# d = pytesseract.image_to_data(im_pil, lang='rus+eng', config='-c preserve_interword_spaces=1x1 --psm 4 --oem 3', output_type=Output.DICT)
# d = pytesseract.image_to_boxes(im_pil, lang='rus+eng', config='-c preserve_interword_spaces=1x1 --psm 12 --oem 3')
# d = pytesseract.image_to_pdf_or_hocr(im_pil, lang='rus', config='-c preserve_interword_spaces=1x1 --psm 4 --oem 3', extension='xml')
d = pytesseract.run_and_get_output(im_pil, lang='rus', config='--psm 4 --oem 3 -c textord_tabfind_find_tables=1 alto')
# d = pytesseract.image_to_string(im_pil, lang='rus+eng', config='-c preserve_interword_spaces=1x1 --psm 4 --oem 3 -c alto=xml')

# h, w, c = image.shape
# for b in d.splitlines():
#     b = b.split(' ')
#     image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
#
# cv2.imshow('img', image)
# cv2.waitKey(0)

# img1 = ImageDraw.Draw(img)
#
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     shape = [(x, y), (x + w, y + h)]
#     img1.rectangle(shape, outline="red")
#
# img.show()

# n_boxes = len(d['block_num'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imshow('img', image)
# cv2.waitKey(0)

print(d)
