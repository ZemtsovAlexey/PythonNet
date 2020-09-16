from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import numpy as np

class DeleteFilesModel:

    def __init__(self):
        self.imageSize = 80
        self.imageChanels = 3
        pass

    def preprocess_image(self, image):
      image = tf.image.decode_jpeg(image, channels=self.imageChanels)
      image = tf.image.resize(image, [self.imageSize, self.imageSize])
      image /= 255.0  # normalize to [0,1] range

      return image

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def remove_files(self, directory_path):
        data_root = pathlib.Path(directory_path)
        all_image_paths = []
        [all_image_paths.extend(data_root.glob('*'))]
        all_image_paths = [str(path) for path in all_image_paths]

        for img_path in all_image_paths:
            os.remove(img_path)

    def test(self, directory_path, name, model):
        falseDir = 'C:/Users/zemtsov/Pictures/deleteFiles/test/false'
        trueDir = 'C:/Users/zemtsov/Pictures/deleteFiles/test/true'

        self.remove_files(falseDir)
        self.remove_files(trueDir)

        data_root = pathlib.Path(directory_path)
        ext = ['png', 'jpg', 'gif']    # Add image formats here
        files = []
        [files.extend(data_root.glob('*/*.' + e)) for e in ext]

        all_image_paths = files
        all_image_paths = [str(path) for path in all_image_paths]

        print('\n')

        i = 0
        t = 0
        f = 0

        for img_path in all_image_paths[:100]:
            oimg = image.load_img(img_path)
            img_tensor = self.get_iamge(img_path)
            predictions = model.predict(img_tensor)

            if (predictions[0][0] < predictions[0][1]):
                oimg.save(trueDir + '/' + name + str(i) + '.jpg', 'JPEG')
                t += 1
            else:
                oimg.save(falseDir + '/' + name + str(i) + '.jpg', 'JPEG')
                f += 1

            i += 1

            print('predict files - ' + str(i) + '   t=' + str(t) + ' f=' + str(f), end='\r')

        print('predict files done - ' + str(i) + '   t=' + str(t) + ' f=' + str(f))

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

        BATCH_SIZE = 5

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
            keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(self.imageSize, self.imageSize, self.imageChanels)),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
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
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255. 

        return img_tensor

    def load_iamge(self, content):
        #img_tensor = tf.image.decode_png(requests.get(url).content, channels=self.imageChanels)
        img_tensor = tf.image.decode_png(content, channels=self.imageChanels)
        img_tensor = tf.image.resize(img_tensor, [self.imageSize, self.imageSize])
        img_tensor /= 255.0
        img_tensor = tf.expand_dims(img_tensor, 0)

        return img_tensor