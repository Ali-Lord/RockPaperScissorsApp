import os
from os.path import isfile, join
import cv2 as cv
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

# Configuration
PIXELS = 300


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

# Turning them into numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Adding a dimension to x
#x_train = x_train[..., np.newaxis]
#x_test = x_test[..., np.newaxis]
#y_train = y_train[..., np.newaxis]
#y_test = y_test[..., np.newaxis]


# Normalizing
x_train = np.array([i / 255.0 for i in x_train])
x_test = np.array([i / 255.0 for i in x_test])

# The model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, padding='valid', activation = 'relu', input_shape=(PIXELS, PIXELS, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='valid', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test, verbose = 1)