from __future__ import absolute_import, division, print_function, unicode_literals
from data.DeleteFilesModel import DeleteFilesModel
from dao.CrmDocuments import CrmDocuments
from tensorflow import keras
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

net = DeleteFilesModel()
documentDao = CrmDocuments()

model = ''
modelName = 'del_model.h5'

if (os.path.exists(modelName)):
    model = keras.models.load_model(modelName)

if (not model):
    (dataset, imageCount) = net.load_dataset('C:/Users/zemtsov/Pictures/deleteFiles/teach')
    model = net.load_model()
    net.train(model, dataset, 10, imageCount)
    model.save(modelName)
    tf.saved_model.save(model, './')
    json_config = model.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(json_config)

ids = documentDao.selectDocumentIds()

falseDir = 'C:/Users/zemtsov/Pictures/deleteFiles/test/false'
trueDir = 'C:/Users/zemtsov/Pictures/deleteFiles/test/true'

net.remove_files(falseDir)
net.remove_files(trueDir)

i = 0
t = 0
f = 0

for id in ids:
    try:
        url = 'http://finite.moedelo.org/bpm/distribution/Rest/Document/' + id
        response = requests.get(url)

        if (response.headers['Content-length'] != '0' and response.headers['Content-type'] == 'image/png'):
            img_tensor = net.load_image(response.content)
            predictions = model.predict(img_tensor)
            img = Image.open(BytesIO(response.content))

            if (predictions[0][0] < predictions[0][1]):
                img.save(trueDir + '/' + str(id) + '.png', 'PNG')
                t += 1
            else:
                img.save(falseDir + '/' + str(id) + '.png', 'PNG')
                f += 1
    finally:        
        i += 1

    print('predict files - ' + str(i) + '   t=' + str(t) + ' f=' + str(f), end='\r')

#tf.keras.models.save_model(model, './')

#tf.saved_model.save(model, './')

#def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#    graph = session.graph
#    with graph.as_default():
#        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#        output_names = output_names or []
#        output_names += [v.op.name for v in tf.global_variables()]
#        input_graph_def = graph.as_graph_def()
#        if clear_devices:
#            for node in input_graph_def.node:
#                node.device = ''
#        frozen_graph = tf.graph_util.convert_variables_to_constants(
#            session, input_graph_def, output_names, freeze_var_names)
#        return frozen_graph

#frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, './', 'xor.pb', as_text=False)

#net.test('C:/Users/zemtsov/Pictures/печать 3/', 'upds', model)