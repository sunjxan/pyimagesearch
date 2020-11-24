import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
# plt.switch_backend("GTK3Agg")

import tensorflow as tf
import tensorflow.keras as keras

class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense0 = keras.layers.Conv2D(filters=10, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')
        self.dense1 = keras.layers.Dense(512, input_dim=1024, activation="sigmoid")
        self.dense2 = keras.layers.Dense(10, input_dim=512, activation="softmax")
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class CNNModel(keras.Model):
  def __init__(self):
    super().__init__()
    # 卷积层1
    self.conv1 = keras.layers.Conv2D(filters=10, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
    # 池化层
    self.pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
    # 卷积层2
    self.conv2 = keras.layers.Conv2D(filters=20, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
    # 展平层
    self.flatten = keras.layers.Flatten()
    # 经过两次池化后得到的矩阵大小为(28/2/2, 28/2/2)
    self.dense1 = keras.layers.Dense(50, input_dim=20*7*7, activation='relu')
    self.dense2 = keras.layers.Dense(10, input_dim=50, activation='softmax')

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    return x

(train_data, train_label), (test_data, test_label) = keras.datasets.cifar10.load_data()
train_data = train_data.astype(np.float32) / 255
train_label = train_label.astype(np.int32)
test_data = test_data.astype(np.float32) / 255
test_label = test_label.astype(np.int32)

LABEL_LOOKUP = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = CNNModel()

model.build(input_shape=(50000, 32, 32, 3))

model.summary()

model.compile(optimizer=keras.optimizers.SGD(5e-3), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_data, train_label, batch_size=32, epochs=10, shuffle=True)

model.evaluate(test_data, test_label, batch_size=32)