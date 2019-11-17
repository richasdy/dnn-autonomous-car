#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:48:00 2019

@author: richasdy
"""

import time
import vrep
import sys
import numpy as np
import matplotlib.pyplot as mpl
import array
from PIL import Image
from datetime import datetime
import pandas as pd 

#Remote API Configuration
vrep.simxFinish(-1) 
clientID=vrep.simxStart ('127.0.0.1',19997,True,True,5000,5)
if clientID!=-1:
    print ("Connected to remote API server") 
    vrep.simxAddStatusbarMessage(clientID,"Program Loaded!",vrep.simx_opmode_oneshot)  
else: 
    print ("Connection not successful")
    sys.exit("Could not connect")

#Get the handle of vision sensor
err,camera = vrep.simxGetObjectHandle(clientID,'golfcar_vision',vrep.simx_opmode_oneshot_wait)

#start Simulator
vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)

#Get the image of vision sensor
err,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_streaming)

#log initialization
driving_log = pd.DataFrame(columns=["center", "right", "left", "sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6", "steering", "thorttle", "speed", "brake"])

while vrep.simxGetConnectionId(clientID) != -1:
    
    # sensor data initialization
    img_name_center = 'data/img/center_'+datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]+'.jpg'
    img_name_right = 'data/img/right_'+datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]+'.jpg'
    img_name_left = 'data/img/left_'+datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]+'.jpg'
    sensor1 = 0
    sensor2 = 0
    sensor3 = 0
    sensor4 = 0
    sensor5 = 0
    sensor6 = 0
    steering = 30
    thorttle = 40
    speed = 10
    brake = 0
    
    # update driving log
    temp_driving_log = {"center" : img_name_center,
                        "right" : img_name_right,
                        "left" : img_name_left,
                        "sensor1" : sensor1,
                        "sensor2" : sensor2,
                        "sensor3" : sensor3,
                        "sensor4" : sensor4,
                        "sensor5" : sensor5,
                        "sensor6" : sensor6,
                        "steering" : steering,
                        "thorttle" : thorttle,
                        "speed" : speed,
                        "brake" :brake }
    driving_log = driving_log.append(temp_driving_log, ignore_index=True)
    driving_log.to_csv("data/driving_log.csv")
    
    print('hi')
    
    
    #image acquisition
    #image center
    #https://programmer.group/v-rep-adding-vision-sensor-and-image-acquisition.html
    err,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_buffer)
    sensorImage = []
    sensorImage = np.array(image,dtype = np.uint8)
    sensorImage.resize([128,128,3])
    img_center = Image.fromarray(sensorImage, 'RGB')
    
    #image right
    img_right = img_center
    
    #image left
    img_left = img_center
    
    # image save
    img_center.save(img_name_center) 
    img_right.save(img_name_right)
    img_left.save(img_name_left)
    
    #make interval 1 second
    time.sleep(1)