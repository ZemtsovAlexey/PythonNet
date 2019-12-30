from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image, ExifTags
import os
import numpy as np
import re

class RotateModel:

    def __init__(self):
        self.imageSize = 100
        self.imageChanels = 1
        self.labels = [0, 180, 270, 90]
        pass

    def preprocess_image(self, img):
      img = tf.image.decode_jpeg(img, channels=self.imageChanels)
      img = tf.image.resize(img, [self.imageSize, self.imageSize])
      img /= 255.0  # normalize to [0,1] range

      return img

    def load_and_preprocess_image(self, path):
        #pil_img = Image.open(path)
        #pil_img = self.reorient_image(pil_img)
        #img  = image.img_to_array(pil_img)
        img = tf.io.read_file(path)

        return self.preprocess_image(img)

    def remove_files(self, directory_path):
        data_root = pathlib.Path(directory_path)
        all_image_paths = []
        [all_image_paths.extend(data_root.glob('*'))]
        all_image_paths = [str(path) for path in all_image_paths]

        for img_path in all_image_paths:
            os.remove(img_path)

    def reorient_image(self, im):
        try:
            image_exif = im._getexif()
            image_orientation = image_exif[274]
            if image_orientation in (2,'2'):
                return im.transpose(Image.FLIP_LEFT_RIGHT)
            elif image_orientation in (3,'3'):
                return im.transpose(Image.ROTATE_180)
            elif image_orientation in (4,'4'):
                return im.transpose(Image.FLIP_TOP_BOTTOM)
            elif image_orientation in (5,'5'):
                return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
            elif image_orientation in (6,'6'):
                return im.transpose(Image.ROTATE_270)
            elif image_orientation in (7,'7'):
                return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
            elif image_orientation in (8,'8'):
                return im.transpose(Image.ROTATE_90)
            else:
                return im
        except (KeyError, AttributeError, TypeError, IndexError):
            return im

    def test(self, directory_path, model):
        save_dir = 'D:/documents types/доки/testRotate'

        #self.remove_files(save_dir)

        data_root = pathlib.Path(directory_path)
        ext = ['png', 'jpg', 'gif']    # Add image formats here
        files = []
        [files.extend(data_root.glob('*/*.' + e)) for e in ext]

        all_image_paths = files
        all_image_paths = [str(path) for path in all_image_paths]

        print('\n')

        i = 0

        for img_path in all_image_paths[0:2000]:
            oimg = image.load_img(img_path)
            oimg = self.reorient_image(oimg)
            img_tensor = self.get_iamge(img_path)
            predictions = model.predict(img_tensor)
            maxIndex, maxValue = max(enumerate(predictions[0]), key=lambda v: v[1])

            #if self.labels[maxIndex] == 90:
            #    oimg = oimg.transpose(Image.ROTATE_90)
            #elif self.labels[maxIndex] == 180:
            #    oimg = oimg.transpose(Image.ROTATE_180)
            #elif self.labels[maxIndex] == 270:
            #    oimg = oimg.transpose(Image.ROTATE_270)

            result = re.search(r'.*\\(.+)\.jpg', img_path)

            oimg.save(save_dir + '/' + str(self.labels[maxIndex]) + '_' + str(i) + '.jpg', 'JPEG')

            i += 1

            print('predict files - ' + str(i), end='\r')

        print('predict files done - ' + str(i))

    def test2(self, directory_path):
        save_dir = 'D:/documents types/доки/testRotate'

        #self.remove_files(save_dir + '/0')
        #self.remove_files(save_dir + '/90')
        #self.remove_files(save_dir + '/180')
        #self.remove_files(save_dir + '/270')

        data_root = pathlib.Path(directory_path)
        ext = ['png', 'jpg', 'gif']    # Add image formats here
        files = []
        [files.extend(data_root.glob('*/*.' + e)) for e in ext]

        all_image_paths = files
        all_image_paths = [str(path) for path in all_image_paths]

        print('\n')

        i = 0

        for img_path in all_image_paths:
            oimg = Image.open(img_path)
            oimg = self.reorient_image(oimg)
            result = re.search(r'(\d+)\\(.+)\.jpg', img_path)

            oimg.save(save_dir + '/' + result.group(1) + '_' + result.group(2) + '.jpg', 'JPEG')

            i += 1

            print('predict files - ' + str(i), end='\r')

        print('predict files done - ' + str(i))

    def load_dataset(self, path):
        data_root = pathlib.Path(path)
        ext = ['png', 'jpg', 'gif'] 
        all_image_paths = []
        [all_image_paths.extend(data_root.glob('*/*.' + e)) for e in ext]
        all_image_paths = [str(path) for path in all_image_paths]
        
        random.shuffle(all_image_paths)

        image_count = len(all_image_paths)

        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        BATCH_SIZE = 10

        ds = image_label_ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return [ds, image_count]

    def load_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(len(self.labels), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, model, ds, epochs = 3, steps_per_epoch=10):
        history = model.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def get_iamge(self, img_path):
        color_mode = 'rgb' if self.imageChanels > 1 else 'grayscale'
        img = image.load_img(img_path, target_size=(self.imageSize, self.imageSize), color_mode=color_mode)
        img = self.reorient_image(img)
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255. 

        return img_tensor

    def load_iamge(self, url):
        from PIL import Image
        import requests
        from io import BytesIO

        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img_tensor = img.resize([self.imageSize, self.imageSize])
        img_tensor = image.img_to_array(img_tensor) 
        img_tensor = np.expand_dims(img_tensor, axis=0)    
        img_tensor /= 255. 

        return img_tensor