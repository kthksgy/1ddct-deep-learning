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


class XceptionModified(keras.Model):
    def __init__(self):
        super(XceptionModified, self).__init__()
        self.bn = keras.layers.normalization.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.mp3_2 = keras.layers.MaxPool2D(3, 2)
        self.gap = keras.layers.GlobalAveragePooling2D()
        self.flatten = keras.layers.Flatten()

        self.entry_conv1 = keras.layers.Conv2D(32, 3, 2, activation='relu')
        self.entry_conv2 = keras.layers.Conv2D(64, 3, 2, activation='relu')

        self.entry_sepconv1 = keras.layers.SeparableConv2D(128, 3, activation='relu')
        self.entry_sepconv2 = keras.layers.SeparableConv2D(128, 3)

        self.entry_conv3 = keras.layers.Conv2D(128, 1, 2)

        self.entry_sepconv3 = keras.layers.SeparableConv2D(256, 3)
        self.entry_sepconv4 = keras.layers.SeparableConv2D(256, 3)

        self.entry_conv4 = keras.layers.Conv2D(256, 1, 2)

        self.entry_sepconv5 = keras.layers.SeparableConv2D(728, 3)
        self.entry_sepconv6 = keras.layers.SeparableConv2D(728, 3)

        self.entry_conv5 = keras.layers.Conv2D(728, 1, 2)

        self.middle_flow = []

        for i in range(8):
            middle_flow.append([
                keras.layers.SeparableConv2D(728, 3, name='middle_sepconv1_%d' % (i + 1)),
                keras.layers.SeparableConv2D(728, 3, name='middle_sepconv2_%d' % (i + 1)),
                keras.layers.SeparableConv2D(728, 3, name='middle_sepconv3_%d' % (i + 1)),
            ])

        self.exit_sepconv1 = keras.layers.SeparableConv2D(728, 3)
        self.exit_sepconv2 = keras.layers.SeparableConv2D(1024, 3)

        self.exit_conv1 = keras.layers.Conv2D(1024, 1, 2)

        self.exit_sepconv3 = keras.layers.SeparableConv2D(1536, 3)
        self.exit_sepconv4 = keras.layers.SeparableConv2D(2048, 3)

        self.fc = keras.layers.Dense(10, activation='softmax')


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
    width, height, channels = info.features['image'].shape

    def cast_image(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    for key in data:
        data[key] = data[key].map(cast_image, num_parallel_calls=16)

    inputs = keras.layers.Input(shape=(32, 32, 3))
    x = inputs

    x = keras.layers.Conv2D(32, 3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(64, 3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    res = keras.layers.Conv2D(128, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.SeparableConv2D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    res = keras.layers.Conv2D(256, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    res = keras.layers.Conv2D(728, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    for i in range(8):
        res = x
        x = keras.layers.SeparableConv2D(728, 3, padding='same', name='middle_sepconv1_%d' % (i + 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv2D(728, 3, padding='same', name='middle_sepconv2_%d' % (i + 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv2D(728, 3, padding='same', name='middle_sepconv3_%d' % (i + 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, res])

    res = keras.layers.Conv2D(1024, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    x = keras.layers.SeparableConv2D(1536, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(2048, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(info.features['label'].num_classes, activation='softmax')(x)

    # model = keras.applications.vgg16.VGG16(
    #     include_top=True,
    #     weights=None,
    #     input_tensor=None,
    #     input_shape=[32, 32, 3],
    #     pooling=None,
    #     classes=info.features['label'].num_classes
    # )
    # model = keras.applications.xception.Xception(
    #     include_top=True,
    #     weights=None,
    #     input_tensor=None,
    #     input_shape=info.features['image'].shape,
    #     pooling=None,
    #     classes=info.features['label'].num_classes
    # )
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    loss_object = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Nadam()

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
        loss = loss_object(label, predictions)

        test_loss(loss)
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
