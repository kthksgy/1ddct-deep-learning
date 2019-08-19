# coding: UTF-8
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from scipy import fftpack
from tensorflow import keras

print('Python Version: ', sys.version)
print('TensorFlow Version: ', tf.__version__)
print('Keras Version: ', keras.__version__)

# tf.debugging.set_log_device_placement(True)


def main():
    # DCTの周波数種類数(-1で一列分の画素数)
    DCT_N = -1
    dataset_name = 'cifar10'
    BATCH_SIZE = 1000
    load_kwargs = {
        'split': None,
        'batch_size': BATCH_SIZE,
        'data_dir': r"W:\dataset",
        'in_memory': True,
        'download': True,
        'as_supervised': True,
        'with_info': True,
        'builder_kwargs': None,
        'download_and_prepare_kwargs': None,
        'as_dataset_kwargs': None
    }
    data, info = tfds.load(dataset_name, **load_kwargs)

    width, height, channels = info.features['image'].shape
    if(DCT_N == -1):
        DCT_N = width

    def cast_image(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    def dct(image, label):
        transposed = tf.transpose(image, (0, 1, 3, 2))
        dcted = tf.signal.dct(transposed, norm='ortho')
        reshaped = tf.reshape(dcted, (BATCH_SIZE, height, -1))
        return reshaped, label

    for key in data:
        data[key] = data[key].map(cast_image, num_parallel_calls=16)
        data[key] = data[key].map(dct, num_parallel_calls=16)

    model = keras.Sequential([
        keras.layers.InputLayer((height, DCT_N * channels)),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(256, 1, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 1, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 1, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(info.features['label'].num_classes, activation='softmax')
    ], name='vgg16_1d')
    model.summary()

    loss_object = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model.compile(optimizer=optimizer, loss=loss_object)

    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image)
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, predictions)

    @tf.function
    def test_step(image, label):
        predictions = model(image)
        t_loss = loss_object(label, predictions)

        test_loss(t_loss)
        test_accuracy(label, predictions)

    file_result = open('result.csv', mode='a')
    file_result.write('epochs,train_loss,train_acc,test_loss,test_acc\n')

    EPOCHS = 1000

    for epoch in range(EPOCHS):
        for image, label in data['train']:
            train_step(image, label)

        for test_image, test_label in data['test']:
            test_step(test_image, test_label)

        template = 'Epoch {:03}, Loss: {:.5f}, Accuracy: {:.4f}, Test Loss: {:.5f}, Test Accuracy: {:.4f}'
        print(
            template.format(
                epoch+1,
                train_loss.result(),
                train_accuracy.result(),
                test_loss.result(),
                test_accuracy.result()
            )
        )
        result_line = '{:03},{:.5f},{:.4f},{:.5f},{:.4f}\n'
        file_result.write(
            result_line.format(
                epoch+1,
                train_loss.result(),
                train_accuracy.result(),
                test_loss.result(),
                test_accuracy.result()
            )
        )
    file_result.close()

    model.save('{}.h5'.format(model.name))


if __name__ == '__main__':
    main()
