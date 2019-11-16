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
import matplotlib.pyplot as mlp
import array
from PIL import Image
from datetime import datetime
import pandas as pd 


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
    

#err,camera = vrep.simxGetObjectHandle(clientID,'golfcar_vision',vrep.simx_opmode_blocking)
err,camera = vrep.simxGetObjectHandle(clientID,'golfcar_vision',vrep.simx_opmode_oneshot_wait)

#
vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)

err,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_streaming)

driving_log = []

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~My Loop Start From Here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

while vrep.simxGetConnectionId(clientID) != -1:
    
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
    
    driving_log = [img_name_center, img_name_right, img_name_left, sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, steering, thorttle, speed, brake ]
#    driving_log.append([img_name_center, img_name_right, img_name_left, sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, steering, thorttle, speed, brake ])
#    rows_list.append(dict1)
#    file = 'data/img/helo.jpg'
    
    print('hi')
    err,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_buffer)
#     err,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_streaming)
#    image_byte_array = np.asarray('b',image)
#    im = Image.frombuffer("RGB", (128,128), image_byte_array, "raw", "RGB", 0, 1)
#    im = Image.fromarray(image_byte_array,"RGB")
#    im = Image.frombuffer("RGBA", (128,128), image_byte_array, "raw", "RGB", 0, 1)
#    im = Image.frombuffer("RGB", (128,128), image, "raw", "RGB", 0, 1)
#    im.show()
    
    w, h = 512, 512
    data = np.zeros((h, w, 4), dtype=np.uint8)

    for i in range(w):
        for j in range(h):
            data[i][j] = [100, 150, 200, 250]
    
    img = Image.fromarray(data, 'RGB')
    # save a image using extension
#    img_url = "/data/img/center_"+timestamp = datetime.utcnow.strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]+".jpg"
    im1 = img.save(img_name_center) 
    
#    break


driving_log = np.asarray(driving_log)
#np.savetxt("data/driving_log.csv", driving_log, delimiter=",") 
pd.DataFrame(driving_log).to_csv("data/driving_log.csv")  
        
#    #Transform the image so it can be displayed using pyplot
#    image_byte_array = array.array('b',image)
##    image_byte_array = np.asarray('b',image)
#    im = Image.frombuffer("RGB", (128,128), image_byte_array, "raw", "RGB", 0, 1)
#    #Update the image
#    plotimg.set_data(im)
#    #Refresh the display
#    mlp.draw()
#    #The mandatory pause ! (or it'll not work)
#    mlp.pause(pause)
    
#    break


    #print(image_byte_array)
#    im = Image.frombuffer("RGB", (256,144), image_byte_array, "raw", "RGB", 0, 1)
#    im.show()
#    x = np.reshape(np.asarray(image), (128, 128, 3))
#    img = np.array(image, dtype = np.uint8)
#    img.resize([resolution[0],resolution[1],3])
#    mlp.imshow(img,origin="lower")
    
    
    #Config sensor
#    sens_read() 
    
    #Initial input sensor 
#    sLeft   = val_s[2]
#    sFront  = val_s[3]
#    sFrontR = val_s[8]
#    sRight  = val_s[9]
    
    
    #Basic Action 
#    fuzzy.input['sensor_left']      = sLeft 
#    fuzzy.input['sensor_leftfront'] = sFront

#    fuzzy.compute()
    
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
    
    
    




    
        
    

    
        
    
    
    
    