import tensorflow as tf
import cv2
import numpy as np

# Configuration
PIXELS = 224
label = ['Rock', 'Paper', 'Scissor']

# The model
model = tf.keras.models.load_model('RockPaperScissorsModel')
model.build(input_shape = (None, PIXELS, PIXELS, 3))
model.summary()

predict_img = cv2.imread('../../data/predict/mom4.jpg')
predict_img = cv2.resize(predict_img, (PIXELS,PIXELS), interpolation = cv2.INTER_AREA)
predict_img = np.array([predict_img / 255.0])
prediction = model.predict(predict_img)

print(label[np.argmax(prediction)])