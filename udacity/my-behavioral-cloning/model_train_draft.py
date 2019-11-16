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

def generator(lines_path, image_path, batch_size=32):
    path = image_path 
    lines = lines_path
    sum_lines = len(lines)
    batch_number=1
    while 1: 
        shuffle(lines)
        for offset in range(0, sum_lines, batch_size):
            batch_samples = lines[offset:offset+batch_size]      
            images = []
            angles = []
            for batch_sample in batch_samples:
                img_choice = np.random.randint(3)
                angle = float(batch_sample[3])
                if angle + bias(batch_number) < np.random.uniform():
                    if img_choice == 0:
                        name =path +batch_sample[1].split('/')[-1]
                        if abs(angle) > 1:
                            angle += 0.25
                        else:
                            angle+=0.18
                    elif img_choice == 1:
                        name = path+batch_sample[0].split('/')[-1]     
                    else:
                        name = path+batch_sample[2].split('/')[-1]
                        if abs(angle) > 1:
                            angle -= 0.25
                        else:
                            angle-=0.18
#                    print('file name : '+name)
                    image = cv2.imread(name)
                    if np.random.randint(10) == 0:
                        images.append(image)
                        angles.append(angle)               
                    if angle!=0.18 and angle!=-0.18 and angle!=0:
                        if np.random.randint(3) == 0:
                            image_new = np.fliplr(image)
                            angle_new = -angle
                            images.append(image_new)
                            angles.append(angle_new)
                        if np.random.randint(3) == 1 or 2:
                            image_new = image_brighten(image)
                            images.append(image_new)
                            angles.append(angle)
                            if np.random.randint(3) == 2:
                                image_new = np.fliplr(image)
                                angle_new = -angle
                                images.append(image_new)
                                angles.append(angle_new)    
                batch_number +=1              
            X_train = np.array(images)
            y_train = np.array(angles)    
            yield shuffle(X_train, y_train)

# Load the images via generator
train_generator = generator(train_samples, image_path)
validation_generator = generator(validation_samples, image_path)

#For visualize lets make some samples that 1024 images for each
x_train_gen, y_train_gen = [], [] 
x_valid_gen, y_valid_gen = [], [] 

x_train_gen, y_train_gen = (next(generator(train_samples, image_path, batch_size=1024)))
x_valid_gen, y_valid_gen = (next(generator(validation_samples, image_path, batch_size=1024)))

#show_steering(y_train_gen, y_valid_gen)

counter = 0
def normal_load(lines_path, image_path):
    global counter
    path = image_path 
    lines = lines_path
    total = len(lines_path)

    shuffle(lines)
    
    images = []
    angles = []
    
    while total >= len(angles):
        for line in lines:
            img_choice = np.random.randint(3)
            angle = float(line[3])
            if angle + bias(counter) < np.random.uniform():
                if img_choice == 0:
                    name =path + line[1].split('/')[-1]

                    if abs(angle) > 1:
                        angle += 0.25
                    else:
                        angle+=0.18
                elif img_choice == 1:
                    name = path + line[0].split('/')[-1]     
                else:
                    name = path + line[2].split('/')[-1]
                    if abs(angle) > 1:
                        angle -= 0.25
                    else:
                        angle-=0.18

                image = cv2.imread(name)
                if np.random.randint(10) == 0:
                    images.append(image)
                    angles.append(angle)


                if angle!=0.18 and angle!=-0.18 and angle!=0:
                    for i in range(3):
                        if np.random.randint(3) == 0:
                            image_new = np.fliplr(image)
                            angle_new = -angle
                            images.append(image_new)
                            angles.append(angle_new)

                        if np.random.randint(3) == 1 or 2:
                            image_new = image_brighten(image)
                            images.append(image_new)
                            angles.append(angle)
                            if np.random.randint(3) == 2:
                                image_new = np.fliplr(image)
                                angle_new = -angle
                                images.append(image_new)
                                angles.append(angle_new) 
            counter +=1                

    X_train = np.array(images)
    y_train = np.array(angles)

    return shuffle(X_train, y_train) 

import time
t0 = time.time()
X_train_normal, y_train_normal = normal_load(train_samples, image_path)
X_valid_normal, y_valid_normal = normal_load(validation_samples, image_path)
t1=time.time()
print("total seconds for loaded: {} sn".format(round(t1-t0)))

print(len(X_train_normal))

#show_steering(y_train_normal, y_valid_normal)

def resize(img):
#    import tensorflow
    import tensorflow.compat.v1 as tensorflow
    tensorflow.disable_v2_behavior()
    return tensorflow.image.resize_images(img, (60, 120))

model = Sequential()
#model = tensorflow.keras.Sequential()

# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((75, 25), (0, 0)),
                     input_shape=(160, 320, 3),
                     data_format="channels_last"))

# Resize the data
model.add(Lambda(resize))

# Normalize the data
model.add(Lambda(lambda x: (x/127.5) - 0.5))

model.add(Conv2D(3, (1, 1), padding='same'))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(Flatten())
model.add(ELU())

model.add(Dense(512))
model.add(Dropout(.2))
model.add(ELU())

model.add(Dense(100))
model.add(Dropout(.5))
model.add(ELU())

model.add(Dense(10))
model.add(Dropout(.5))
model.add(ELU())

model.add(Dense(1))

adam = Adam(lr=1e-5)
model.compile(optimizer= adam, loss="mse", metrics=['accuracy'])

#Showing model and params, it is a heavy network with nearly 2.2 bilion params
model.summary()

# error
#file = 'model_generator.h5'
#earlystopper = EarlyStopping(patience=5, verbose=1)
#checkpointer = ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only=True)
#model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
#                    validation_data=validation_generator,
#                    validation_steps=len(validation_samples), epochs = 1)
#model.save(file)



earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model_nor.h5', monitor='val_loss', verbose=1, save_best_only=True)
#results = model.fit(X_train_normal, y_train_normal,
#                    validation_data=(X_valid_normal, y_valid_normal),
#                    batch_size=32, epochs=50, verbose=1)
results = model.fit(X_train_normal, y_train_normal,
                   batch_size=32, epochs=1, verbose=1)
#tensorflow.keras.Sequential.save()
#model.save('model_nor.h5')
model.save_weights('model_nor_weight.h5')

#save file
text_file = open("model_nor.json", "w")
n = text_file.write(model.to_json())
text_file.close()


##For fun lets train raw data
#earlystopper = EarlyStopping(patience=5, verbose=1)
#checkpointer = ModelCheckpoint('model_row.h5', monitor='val_loss', verbose=1, save_best_only=True)
##results = model.fit(X_train_s, y_train_s,
##                    validation_data=(X_valid_s, y_valid_s),
##                    batch_size=32, epochs=30, verbose=1)
#results = model.fit(X_train_s, y_train_s,
#                    validation_data=(X_valid_s, y_valid_s),
#                    batch_size=32, epochs=1, verbose=1)
#model.save('model_row.h5')