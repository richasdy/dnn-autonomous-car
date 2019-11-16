#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:37:06 2019

@author: richasdy
"""
import cv2
import numpy as np
import csv
import tensorflow
import pandas as pd

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D 
from keras.layers import Lambda, Cropping2D, Dropout, ELU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

#load lines in csv file to read the images 
def load_csv(file):
    lines = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines[1:]

# generally load images in the file 
def load_images(lines_path, image_path):
    path = image_path 
    lines = lines_path
    images = []
    angles = []
    
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = path+filename
            image = cv2.imread(current_path)
            images.append(image)
            angle = float(line[3])
            if i == 0:
                angles.append(angle)
            elif i == 1:
                angles.append(angle + 0.20)
            else:
                angles.append(angle - 0.20)
    X_train = np.array(images)
    y_train = np.array(angles) 
    return X_train, y_train

# Load Images and Split
    
file = './data-dummy/driving_log.csv'
image_path = './data-dummy/IMG/'

#load image names in the csv file
lines_path = load_csv(file)

#split not the data just their names
# We don't need to test images because it is a regression problem not classification.
train_samples, validation_samples = train_test_split(lines_path, shuffle=True, test_size=0.2)
X_train, y_train = load_images(train_samples, image_path)
X_valid, y_valid = load_images(validation_samples, image_path)

#Check Train and Valid data samples
assert len(X_train) == len(y_train), "X_train {} and y_train {} are not equal".format(len(X_train), len(y_train))
assert len(X_valid) == len(y_valid), "X_valid {} and y_valid {} are not equal".format(len(X_valid), len(y_valid))
print('Total Train samples: {}\nTotal Valid samples: {}'.format(len(X_train), len(X_valid)))

#Shuffle train and validation sets - also keras have a shuffle attribute in training
X_train_s, y_train_s = shuffle(X_train, y_train)
X_valid_s, y_valid_s= shuffle(X_valid, y_valid)

# Augmention Phase
def image_brighten(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    default_bias = 0.25
    brightness = default_bias + np.random.uniform() 
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def bias(n):
    return 1. / (n + 1.)

def bias(batch_number):
    '''take batch number and return a bias term'''
    return 1. / (batch_number + 1.)

def resize(img):
#    import tensorflow
    import tensorflow.compat.v1 as tensorflow
    tensorflow.disable_v2_behavior()
    return tensorflow.image.resize_images(img, (60, 120))

model = Sequential()
#model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160, 320, 3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()
checkpoint = ModelCheckpoint('model_train_naoki.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only='true',
                                 mode='auto')
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
results = model.fit(X_train_s, y_train_s, batch_size=32, epochs=1, verbose=1)

#model.fit_generator(batch_generator(image_path, X_train, y_train, 40, True),
#                        20000,
#                        10,
#                        max_q_size=1,
#                        validation_data=batch_generator(image_path, X_valid, y_valid, 40, False),
#                        nb_val_samples=len(X_valid),
#                        callbacks=[checkpoint],
#                        verbose=1)

# =============================================================================
# model = Sequential()
# #model = tensorflow.keras.Sequential()
# 
# # Crop 70 pixels from the top of the image and 25 from the bottom
# model.add(Cropping2D(cropping=((75, 25), (0, 0)),
#                      input_shape=(160, 320, 3),
#                      data_format="channels_last"))
# 
# # Resize the data
# model.add(Lambda(resize))
# 
# # Normalize the data
# model.add(Lambda(lambda x: (x/127.5) - 0.5))
# 
# model.add(Conv2D(3, (1, 1), padding='same'))
# model.add(ELU())
# 
# model.add(BatchNormalization())
# model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same"))
# model.add(ELU())
# 
# model.add(BatchNormalization())
# model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
# model.add(ELU())
# 
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
# model.add(ELU())
# 
# model.add(BatchNormalization())
# model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
# model.add(ELU())
# 
# model.add(Flatten())
# model.add(ELU())
# 
# model.add(Dense(512))
# model.add(Dropout(.2))
# model.add(ELU())
# 
# model.add(Dense(100))
# model.add(Dropout(.5))
# model.add(ELU())
# 
# model.add(Dense(10))
# model.add(Dropout(.5))
# model.add(ELU())
# 
# model.add(Dense(1))
# 
# adam = Adam(lr=1e-5)
# model.compile(optimizer= adam, loss="mse", metrics=['accuracy'])
# 
# #Showing model and params, it is a heavy network with nearly 2.2 bilion params
# model.summary()
# 
# earlystopper = EarlyStopping(patience=5, verbose=1)
# checkpointer = ModelCheckpoint('model_train_naoki.h5', monitor='val_loss', verbose=1, save_best_only=True)
# results = model.fit(X_train_s, y_train_s, batch_size=32, epochs=1, verbose=1)
# 
# =============================================================================
model.save('model_train_naoki.h5')

print(model.evaluate(X_valid_s, y_valid_s))

print(model.predict([[X_train_s[0]]]))
