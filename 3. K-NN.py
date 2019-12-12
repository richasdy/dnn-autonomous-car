# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:22:34 2018

@author: darfyma
"""

#Support Vector Machine (SVM)
import numpy as np
import cv2
from PIL import Image
#from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
#from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
import warnings
warnings.filterwarnings('ignore')

X = np.zeros([1,102400])
y = np.zeros([1,1])
angka = range (0,6)
nim = range (1,14)
ulang = range (1,6)

for eachAng in angka:
    for eachNim in nim:
        for eachUl in ulang:
            try:
                #img = cv2.imread("Database/"+str(i)+"-"+str(j)+"-"+str(k)+".png",0)
                imgFilePath = "database/"+str(eachAng)+"-"+str(eachNim)+"-"+str(eachUl)+".png"
                ei = Image.open(imgFilePath)
                eiar = np.array(ei)
                eiar = np.expand_dims(eiar.flatten(),axis=0)
                X = np.append(X,eiar,axis=0)
                y = np.append(y,eachAng)
            except Exception as e:
                pass
            
X=np.delete(X,0,0)
y=np.delete(y,0,0)
X = X / 255

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=1)

#3_KNN Parameter 

clf = neighbors.KNeighborsClassifier(2)


clf.fit(X,y)
y_knn = clf.predict(X_test)
print(y_knn)
print(y_test)
print(classification_report(y_test,y_knn))


# global variables
bg = None
#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return
    
# compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)
#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)   
    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                    threshold,
                    255,
                    cv2.THRESH_BINARY)[1]
    
    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                         cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_SIMPLE)
    
    
    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
camera = cv2.VideoCapture(1)

# Check if camera opened successfully
if (camera.isOpened()== False):
    print("Error opening video stream or file")
# initialize weight for running average
aWeight = 0.5
# initialize num of frames
num_frames = 0

# Read until video is completed
while(camera.isOpened()):
# get the current frame
    grabbed, frame = camera.read()
    if not grabbed:
        break
    # resize the frame
    #frame = cv2.resize(frame, (640,480))
    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)   
    
    # clone the frame
    clone = frame.copy()
    # get the height and width of the frame
    (height, width) = frame.shape[:2]
    # region of interest (ROI) coordinates
    side = 320
    top = 10
    bottom = top+side
    left = width - 10
    right = left - side
    # get the ROI
    roi = frame[top:bottom, right:left]
    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # to get the background, keep looking till a threshold is reached
    # so that our running average model gets calibrated
    if num_frames < 30:
        run_avg(gray, aWeight)
        # increment the number of frames
        num_frames += 1
    else:
        # segment the hand region
        hand = segment(gray)
        # check whether hand region is segmented
        if hand is not None:
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, segmented) = hand
            
            # draw the segmented region and display the frame
            cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
            cv2.imshow("Thesholded", thresholded)
            output = np.array(thresholded)
            output = np.expand_dims(output.flatten(),axis=0)
            output = output / 255
            res = clf.predict(output)
            print(res)
            
            cv2.putText(clone,str(res[0]),(10,450),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)
    # draw the segmented hand
    cv2.rectangle(clone,(left, top),(right, bottom),(0,255,0),2)
    
    # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)
    # observe the keypress by the user
    #keypress = cv2.waitKey(1) & 0xFF

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    
    # if the user pressed "q", then stop looping
    if k== ord("q"):
        break
        
    # if the user pressed "r", then reset background
    if k == ord("r"):
        bg = None
        num_frames = 0


camera.release()
cv2.destroyAllWindows()    
    
    