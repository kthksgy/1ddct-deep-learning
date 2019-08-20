# coding: UTF-8
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras

print('Python Version: ', sys.version)
print('TensorFlow Version: ', tf.__version__)
print('Keras Version: ', keras.__version__)

# tf.debugging.set_log_device_placement(True)


def main():
    dataset_name = 'cifar10'
    load_kwargs = {
        'split': None,
        'batch_size': 1000,
        'data_dir': r"~/.datasets",
        'in_memory': True,
        'download': True,
        'as_supervised': True,
        'with_info': True,
        'builder_kwargs': None,
        'download_and_prepare_kwargs': None,
        'as_dataset_kwargs': None
    }
    data, info = tfds.load(dataset_name, **load_kwargs)

    def cast_image(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    for key in data:
        data[key] = data[key].map(cast_image, num_parallel_calls=16)

    model = keras.applications.vgg16.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=[32, 32, 3],
        pooling=None,
        classes=info.features['label'].num_classes
    )
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

        template = 'Epoch {:0%d}, Loss: {:.5f}, Accuracy: {:.4f}, Test Loss: {:.5f}, Test Accuracy: {:.4f}' % len(str(EPOCHS))
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result(),
                test_loss.result(),
                test_accuracy.result()
            )
        )
        result_line = '{:0%d},{:.5f},{:.4f},{:.5f},{:.4f}\n' % len(str(EPOCHS))
        file_result.write(
            result_line.format(
                epoch + 1,
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
