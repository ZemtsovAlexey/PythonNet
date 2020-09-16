from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pathlib
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


class TextModel:

    def __init__(self):
        self.imageSize = 22
        self.imageChanel = 1
        # self.labels = list('0123456789abcdefghijklmnopqrstuvwxyz')
        self.labels = list('0123456789абвгдежзийклмнопрстуфхцчшщъыьэюяё')
        pass

    def pre_process_image(self, image):
        image = tf.image.decode_jpeg(image, channels=self.imageChanel)
        image = tf.image.resize(image, [self.imageSize, self.imageSize])
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.pre_process_image(image)

    def remove_files(self, directory_path):
        data_root = pathlib.Path(directory_path)
        all_image_paths = []
        [all_image_paths.extend(data_root.glob('*'))]
        all_image_paths = [str(path) for path in all_image_paths]

        for img_path in all_image_paths:
            os.remove(img_path)

    def test(self, directory_path, model):
        dir = 'C:/Users/zemtsov/Pictures/typingTest'

        self.remove_files(dir)

        data_root = pathlib.Path(directory_path)
        ext = ['png', 'jpg', 'gif']  # Add image formats here
        files = []
        [files.extend(data_root.glob('*/*.' + e)) for e in ext]

        all_image_paths = files
        all_image_paths = [str(path) for path in all_image_paths]

        print('\n')

        i = 0

        for img_path in all_image_paths[1000:2000]:
            oimg = image.load_img(img_path)
            img_tensor = self.get_image(img_path)
            predictions = model.predict(img_tensor)

            # maxIndex = np.where(predictions[0]==x.max())[0]
            maxIndex, maxValue = max(enumerate(predictions[0]), key=lambda v: v[1])

            oimg.save(dir + '/' + self.labels[maxIndex] + '_' + str(i) + '.jpg', 'JPEG')

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
        print(label_names)
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
            keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                                input_shape=(self.imageSize, self.imageSize, self.imageChanel)),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            # keras.layers.Dropout(0.2),
            # keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            # keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(60, activation='relu'),
            keras.layers.Dense(len(self.labels), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, model, ds, epochs=3, steps_per_epoch=10):
        history = model.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def predict(self, model, img):
        batch = np.empty(shape=(len(img), self.imageSize, self.imageSize, self.imageChanel))

        for i, im in enumerate(img):
            img_tensor = im.convert('L')
            img_tensor = img_tensor.resize([self.imageSize, self.imageSize])
            img_tensor = image.img_to_array(img_tensor)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            batch[i] = img_tensor

        predictions = model.predict(batch)

        result = list()

        for predict in predictions:
            max_index, max_value = max(enumerate(predict), key=lambda v: v[1])
            result.append(self.labels[max_index])

        return result

    def get_image(self, img_path):
        color_mode = 'rgb' if self.imageChanel > 1 else 'grayscale'
        img = image.load_img(img_path, target_size=(self.imageSize, self.imageSize), color_mode=color_mode)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        return img_tensor

    def load_image(self, url):
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
