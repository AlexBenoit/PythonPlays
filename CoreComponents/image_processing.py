from __future__ import print_function
import argparse
import cv2
import sys
import numpy as np
from random import randint
#import imutils

import grabScreen

from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT, MODEL_PATH

def process_image(image):
    print("processing image")

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
     
    return tracker

if __name__ == "__main__":

    #backSub = cv2.createBackgroundSubtractorMOG2()

    backSub = cv2.createBackgroundSubtractorKNN()

    ## Select boxes
    bboxes = []
    colors = [] 

    screen = grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', screen)
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break
 
    print('Selected bounding boxes {}'.format(bboxes))

    while True:
        screen = grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
        #screen = cv2.resize(screen, (int(RECORDING_WIDTH/4), int(RECORDING_HEIGHT/4)))
        #screen = cv2.Canny(screen, 100, 150) # edge detection
        #thresh = cv2.threshold(screen, 200, 255, cv2.THRESH_BINARY_INV)[1]

        #cv2.imshow("window", thresh) # Window showing what is captured
        #cv2.waitKey(1)
    
        #screen = backSub.apply(screen)
        #thresh = cv2.threshold(screen, 220, 255, cv2.THRESH_BINARY_INV)[1]
    
        #cv2.rectangle(screen, (10, 2), (100,20), (255,255,255), -1)
        #cv2.putText(screen, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        #cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)
        #output = image.copy()
 
        # loop over the contours
        #for c in cnts:
	        # draw each contour on the output image with a 3px thick purple
	        # outline, then display the output contours one at a time
	        #cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	        #cv2.imshow("Contours", output)
	        #cv2.waitKey(0)

        window = cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", screen)
        #cv2.imshow('FG Mask', fgMask)
    
        keyboard = cv2.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break