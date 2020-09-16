import tensorflow as tf
import tensorflow_datasets as tfds
import os

FILE_NAMES = ['Payroll', 'CallbackClient', 'DocumentRequest', 'LoadingUnloading1c', 'QuestionToAccountant']
parent_dir = '/mnt/c/rrr/test/'

def labeler(example, index):
  return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name + '.txt'))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)




BUFFER_SIZE = 1000
BATCH_SIZE = 2
TAKE_SIZE = 10

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)




tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
vocab_size


encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

# example_text = next(iter(all_labeled_data))[0].numpy()
# encoded_example = encoder.encode(example_text)



def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)



vocab_size += 1


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 5))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))

# Один или более плотных слоев.
# Отредактируйте список в строке `for` чтобы поэкспериментировать с размером слоев.
for units in [32]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Выходной слой. Первый аргумент - число меток.
model.add(tf.keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_data, epochs=1, validation_data=test_data)


eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))