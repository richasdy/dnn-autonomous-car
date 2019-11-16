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
from tensorflow.keras.models import load_model

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D 
from keras.layers import Lambda, Cropping2D, Dropout, ELU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

model = load_model('model_train_naoki.h5')
#model = load_model('model_train_turki.h5')

#model.evaluate(X_train_s, y_train_s)

print(model.predict([[X_train_s[0]]]))