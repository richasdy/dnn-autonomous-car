# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:02:59 2019
@author: HambaAllah
Dibuat dalam rangka Syarat kelulusan matakuliah Tesis Magister instrumentasi dan Kontrol 
Bismillah
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
import csv
import cv2


MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~My Program Start From Here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
#Remote API Configuration
vrep.simxFinish(-1) 
clientID=vrep.simxStart ('127.0.0.1',19997,True,True,5000,5)
if clientID!=-1:
    print ("Connected to remote API server") 
    vrep.simxAddStatusbarMessage(clientID,"Program Loaded!",vrep.simx_opmode_oneshot)  
else: 
    print ("Connection not successful")
    sys.exit("Could not connect")
    
err,camera = vrep.simxGetObjectHandle(clientID,'golfcar_vision',vrep.simx_opmode_oneshot_wait)

vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)

err,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_streaming)

images = []    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~My Loop Start From Here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

while vrep.simxGetConnectionId(clientID) != -1:
    
    
    
    # sensor data initialization
    img_name_center = 'data/img/center_'+datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]+'.jpg'
    img_name_right = 'data/img/right_'+datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]+'.jpg'
    img_name_left = 'data/img/left_'+datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]+'.jpg'
    
    img_center = []
    img_right = []
    img_left = []
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
    
    
    
    #sensor data acquisition
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
    
    #sensor1
    sensor1 = 0
    
    #sensor2
    sensor2 = 0
    
    #sensor3
    sensor3 = 0
    
    #sensor4
    sensor4 = 0
    
    #sensor5
    sensor5 = 0
    
    #sensor6
    sensor6 = 0
    
    #steering
    steering = 30
    
    #thorttle
    thorttle = 40
    
    #speed
    speed = 10
    
    #brake
    brake = 0
    
    
    
    
    # prediction
    images = images.append(sensorImage)
    images = images.append(sensorImage)
    images = images.append(sensorImage)
    images = np.array(images)    
    
    model = load_model('model_train_turki_putra.h5')
    
    steering = model.predict([[images[0]]])
    
    #inspire by naoki
    if speed > speed_limit:
        speed_limit = MIN_SPEED  # slow down
    else:
        speed_limit = MAX_SPEED
    
    throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

    print('{} {} {}'.format(steering, throttle, speed))
    
    send_control(steering_angle, throttle)

    
    #fuzzy value     
    #Initial input Output for Motor + You can add Braitenberg for Turning Robot 
#    vl = (fuzzy.output['outputL']) 
#    vr = (fuzzy.output['outputR'])   
    
    #fuzzy + braitenberg for additional turning
#    vBrLeft=vl
#    vBrRight=vr
#    for i in range(0,4):
#        vBrLeft=vBrLeft+(braitenbergL[i]*(1-val_s[i+4]))
#        vBrRight=vBrRight+(braitenbergR[i]*(1-val_s[i+4]))
#
#    forward(5+vBrLeft,5+vBrRight)
    
    
    




    
        
    

    
        
    
    
    
    