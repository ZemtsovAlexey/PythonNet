from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.preprocessing import image
import pathlib
import matplotlib.pyplot as plt
import os
import numpy as np
import re

labels = ['invoices', 'statements', 'trash', 'upds', 'waybills']

def remove_files(directory_path):
    data_root = pathlib.Path(directory_path)
    all_image_paths = []
    [all_image_paths.extend(data_root.glob('*'))]
    all_image_paths = [str(path) for path in all_image_paths]
    
    for img_path in all_image_paths:
        os.remove(img_path)

def test(directory_path, save_path):
    remove_files('D:/documents types/доки/teach/invoices')
    remove_files('D:/documents types/доки/teach/statements')
    remove_files('D:/documents types/доки/teach/upds')
    remove_files('D:/documents types/доки/teach/waybills')
    
    dirs = []
    dirs.append('D:/documents types/доки/доки/statements')
    dirs.append('D:/documents types/доки/доки/invoices')
    dirs.append('D:/documents types/доки/доки/upds')
    dirs.append('D:/documents types/доки/доки/waybills')

    for dir in dirs:
        data_root = pathlib.Path(dir)
        ext = ['png', 'jpg']    # Add image formats here
        files = [(dir + '/' + item.name + '/' + (os.listdir(dir + '/' + item.name))[0]) for item in data_root.glob('*/') if item.is_dir()]
    
        print('\n')
    
        i = 0
        ab = re.compile("png|jpg")
    
        for file in files[:500]:
            m = ab.search(file)

            if not m:
                continue

            oimg = image.load_img(file)
            result = re.search(r'\/(\w+)\/\d+\/.+\.', file)
            oimg.save(save_path + '/' + result.group(1) + '/'  + str(i) + '.jpg', 'JPEG')
    
            i += 1
    
            print('predict files - ' + str(i), end='\r')
    
        print('predict files done - ' + str(i))

test('D:/documents types/доки/доки/', 'D:/documents types/доки/teach')