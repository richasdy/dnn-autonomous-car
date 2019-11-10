#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:05:29 2019

@author: richasdy
@ref: https://www.youtube.com/watch?v=BBwEF6WBUQs
@ref: https://github.com/hamuchiwa/AutoRCCar
@ref: https://zhengludwig.wordpress.com/projects/self-driving-rc-car/
"""

import cv2
import numpy as np
import glob
import sys
# import matplotlib.pyplot as plt
# %matplotlib inline

print ("OpenCV:",  cv2.__version__)
print ("Numpy : ", np.__version__)
print ("Python:",  sys.version)

# load training data
dim = 240*320
X = np.empty((0, dim))
y = np.empty((0, 4))
training_data = glob.glob('auto-rccar-data.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
        train = data['train']
        train_labels = data['train_labels']
    X = np.vstack((X, train))
    y = np.vstack((y, train_labels))

print ('Image array shape: ', X.shape)
print ('Label array shape: ', y.shape)


# load model
model = cv2.ml.ANN_MLP_load('auto-rccar-model.xml')

# predict
ret, resp = model.predict(X)
print (len(resp))
resp.argmax(-1)
print(resp.argmax(-1))
