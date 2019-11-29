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
from tensorflow.keras.models import load_model


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
    
#image handler
err_handler_camera_c,handler_camera_c = vrep.simxGetObjectHandle(clientID,'golfcar_vision_c',vrep.simx_opmode_oneshot_wait)
err_handler_camera_l,handler_camera_l = vrep.simxGetObjectHandle(clientID,'golfcar_vision_l',vrep.simx_opmode_oneshot_wait)
err_handler_camera_r,handler_camera_r = vrep.simxGetObjectHandle(clientID,'golfcar_vision_r',vrep.simx_opmode_oneshot_wait)

#sensor handler
#err_handler_sensor1,handler_sensor1 = vrep.simxGetObjectHandle(clientID,'sensor1',vrep.simx_opmode_oneshot_wait)
#err_handler_sensor2,handler_sensor2 = vrep.simxGetObjectHandle(clientID,'sensor2',vrep.simx_opmode_oneshot_wait)
#err_handler_sensor3,handler_sensor3 = vrep.simxGetObjectHandle(clientID,'sensor3',vrep.simx_opmode_oneshot_wait)
#err_handler_sensor4,handler_sensor4 = vrep.simxGetObjectHandle(clientID,'sensor4',vrep.simx_opmode_oneshot_wait)
#err_handler_sensor5,handler_sensor5 = vrep.simxGetObjectHandle(clientID,'sensor5',vrep.simx_opmode_oneshot_wait)
#err_handler_sensor6,handler_sensor6 = vrep.simxGetObjectHandle(clientID,'sensor6',vrep.simx_opmode_oneshot_wait)

#state handler
#err_handler_steering,handler_steering = vrep.simxGetObjectHandle(clientID,'steering',vrep.simx_opmode_oneshot_wait)
#err_handler_thorttle,handler_thorttle = vrep.simxGetObjectHandle(clientID,'thorttle',vrep.simx_opmode_oneshot_wait)
#err_handler_speed,handler_speed = vrep.simxGetObjectHandle(clientID,'speed',vrep.simx_opmode_oneshot_wait)
#err_handler_brake,handler_brake = vrep.simxGetObjectHandle(clientID,'brake',vrep.simx_opmode_oneshot_wait)

vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)

#get image
err_image_c,resolution_image_c,image_c = vrep.simxGetVisionSensorImage(clientID,handler_camera_c,0,vrep.simx_opmode_streaming)
err_image_l,resolution_image_c,image_l = vrep.simxGetVisionSensorImage(clientID,handler_camera_l,0,vrep.simx_opmode_streaming)
err_image_r,resolution_image_c,image_r = vrep.simxGetVisionSensorImage(clientID,handler_camera_r,0,vrep.simx_opmode_streaming)

#get sensor
err_signal1,signal1=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor1',vrep.simx_opmode_streaming) #front
err_signal2,signal2=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor2',vrep.simx_opmode_streaming) #front_right
err_signal3,signal3=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor3',vrep.simx_opmode_streaming) #front_left
err_signal4,signal4=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor4',vrep.simx_opmode_streaming) #rear_left
err_signal5,signal5=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor5',vrep.simx_opmode_streaming) #rear_right
err_signal6,signal6=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor6',vrep.simx_opmode_streaming) #rear

#get state
err_steering,steering=vrep.simxGetFloatSignal(clientID,'handler_steering',vrep.simx_opmode_streaming)
err_thorttle,thorttle=vrep.simxGetFloatSignal(clientID,'handler_thortle',vrep.simx_opmode_streaming)
err_speed,speed=vrep.simxGetFloatSignal(clientID,'handler_speed',vrep.simx_opmode_streaming)
err_brake,brake=vrep.simxGetIntegerSignal(clientID,'handler_brake',vrep.simx_opmode_streaming)

images = []    

model = load_model('model_train_turki_putra_1000.h5')

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
    steering = 0
    thorttle = 0
    speed = 0
    brake = 0
    
    
    
    #sensor data acquisition
    #image acquisition
    #image center
    #https://programmer.group/v-rep-adding-vision-sensor-and-image-acquisition.html
    err_c,resolution_c,image_c = vrep.simxGetVisionSensorImage(clientID,handler_camera_c,0,vrep.simx_opmode_buffer)
    sensorImage_c = []
    sensorImage_c = np.array(image_c,dtype = np.uint8)
    sensorImage_c.resize([256,512,3])
#    sensorImage_c.resize([128,128,3])
    img_center = Image.fromarray(sensorImage_c, 'RGB')
#    img_center = Image.open(BytesIO(base64.b64decode(img_center)))
    
    #image left
    err_l,resolution_l,image_l = vrep.simxGetVisionSensorImage(clientID,handler_camera_l,0,vrep.simx_opmode_buffer)
    sensorImage_l = []
    sensorImage_l = np.array(image_c,dtype = np.uint8)
    sensorImage_l.resize([256,512,3])
#    sensorImage_l.resize([128,128,3])
    img_left = Image.fromarray(sensorImage_l, 'RGB')
#    img_left = Image.open(BytesIO(base64.b64decode(img_left)))
    
    #image right
    err_r,resolution_r,image_r = vrep.simxGetVisionSensorImage(clientID,handler_camera_r,0,vrep.simx_opmode_buffer)
    sensorImage_r = []
    sensorImage_r = np.array(image_r,dtype = np.uint8)
    sensorImage_r.resize([256,512,3])
#    sensorImage_r.resize([128,128,3])
    img_right = Image.fromarray(sensorImage_r, 'RGB')
#    img_right = Image.open(BytesIO(base64.b64decode(img_right)))
    
     #sensor1
    err_signal1,signal1=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor1',vrep.simx_opmode_buffer)
    
    #sensor2
    err_signal2,signal2=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor2',vrep.simx_opmode_buffer)
    
    #sensor3
    err_signal3,signal3=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor3',vrep.simx_opmode_buffer)
    
    #sensor4
    err_signal4,signal4=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor4',vrep.simx_opmode_buffer)
    
    #sensor5
    err_signal5,signal5=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor5',vrep.simx_opmode_buffer)
    
    #sensor6
    err_signal6,signal6=vrep.simxGetFloatSignal(clientID,'golfcar_ultra_sensor6',vrep.simx_opmode_buffer)
    
    #steering
    err_steering,steering=vrep.simxGetFloatSignal(clientID,'handler_steering',vrep.simx_opmode_buffer)
    
    #thorttle
    err_thorttle,thorttle=vrep.simxGetFloatSignal(clientID,'handler_thortle',vrep.simx_opmode_buffer)
    
    #speed
    err_speed,speed=vrep.simxGetFloatSignal(clientID,'handler_speed',vrep.simx_opmode_buffer)
    
    #brake
    err_brake,brake=vrep.simxGetFloatSignal(clientID,'handler_brake',vrep.simx_opmode_buffer)
    
    
    
    
    # prediction
    
    # from turki training
    images = [sensorImage_c, sensorImage_l, sensorImage_r]
    images = np.array(images)
    
    # only use center image to drive
    steering = model.predict([[images[0]]])
    
    #inspired by naoki
#    if speed > speed_limit:
#        speed_limit = MIN_SPEED  # slow down
#    else:
#        speed_limit = MAX_SPEED
#    throttle = 1.0 - steering**2 - (speed/speed_limit)**2
    
    throttle = 7
    
    print('{} {} {} {} {}'.format(datetime.utcnow(), steering, throttle, speed, brake))
    
    
    
    # send control
    vrep.simxSetFloatSignal(clientID,'target_steering',steering ,vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID,'target_throttle',throttle ,vrep.simx_opmode_oneshot)
#    vrep.simxSetFloatSignal(clientID,'target_speed',speed ,vrep.simx_opmode_oneshot)
#    vrep.simxSetFloatSignal(clientID,'target_brake',brake ,vrep.simx_opmode_oneshot)
    
    
    
    