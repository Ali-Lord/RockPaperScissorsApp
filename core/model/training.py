import os
from os.path import isfile, join
import cv2 as cv
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
#import matplotlib.pyplot as plt

# Configuration
PIXELS = 224
random.seed(1000)

# Images
x_train = []
x_test = []

# labels - 0 for rock, 1 for paper and 2 for scissor.
y_train = []
y_test = []

# labels name
yNames = ['Rock', 'Paper', 'Scissor']

# Read directories
rockDirPathList_train = [f for f in os.listdir('../../data/train/rock')]
paperDirPathList_train = [f for f in os.listdir('../../data/train/paper')]
scissorDirPathList_train = [f for f in os.listdir('../../data/train/scissor')]
rockDirPathList_test = [f for f in os.listdir('../../data/test/rock')]
paperDirPathList_test = [f for f in os.listdir('../../data/test/paper')]
scissorDirPathList_test = [f for f in os.listdir('../../data/test/scissor')]

# Read images
for image_name in rockDirPathList_train:
    img = cv.imread('../../data/train/rock/' + image_name)
    x_train.append(cv.resize(img, (PIXELS,PIXELS), interpolation = cv.INTER_AREA))
    y_train.append(0)

for image_name in paperDirPathList_train:
    img = cv.imread('../../data/train/paper/' + image_name)
    x_train.append(cv.resize(img, (PIXELS,PIXELS), interpolation = cv.INTER_AREA))
    y_train.append(1)
    
for image_name in scissorDirPathList_train:
    img = cv.imread('../../data/train/scissor/' + image_name)
    x_train.append(cv.resize(img, (PIXELS,PIXELS), interpolation = cv.INTER_AREA))
    y_train.append(2)
    
for image_name in rockDirPathList_test:
    img = cv.imread('../../data/test/rock/' + image_name)
    x_test.append(cv.resize(img, (PIXELS,PIXELS), interpolation = cv.INTER_AREA))
    y_test.append(0)

for image_name in paperDirPathList_test:
    img = cv.imread('../../data/test/paper/' + image_name)
    x_test.append(cv.resize(img, (PIXELS,PIXELS), interpolation = cv.INTER_AREA))
    y_test.append(1)
    
for image_name in scissorDirPathList_test:
    img = cv.imread('../../data/test/scissor/' + image_name)
    x_test.append(cv.resize(img, (PIXELS,PIXELS), interpolation = cv.INTER_AREA))
    y_test.append(2)

# Shuffle
train_z = list(zip(x_train, y_train))
random.shuffle(train_z)
x_train, y_train = zip(*train_z)

test_z = list(zip(x_test, y_test))
random.shuffle(test_z)
x_test, y_test = zip(*test_z)

# Turning them into numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# Normalizing
x_train = np.array([i / 255.0 for i in x_train])
x_test = np.array([i / 255.0 for i in x_test])

# The model
base_model = tf.keras.applications.MobileNetV2(input_shape=(PIXELS, PIXELS, 3), include_top=False, weights='imagenet')
base_model.trainable = False
average_layer = tf.keras.layers.GlobalAveragePooling2D()

model = tf.keras.Sequential([
    base_model,
    average_layer,
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150, batch_size = 2)
model.evaluate(x_test, y_test, verbose = 1)

model.save('RockPaperScissorsModel') 