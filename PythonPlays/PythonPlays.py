#!/usr/bin/python

import numpy as np
import cv2
import time
import random
import keyboard
import keyInputs
import numpy
import imageProcessing as imgProc
import grabScreen
from textAnalyzer import TextAnalyzer
from FrameComparator import FrameComparator
import tensorflowNN
import smashMeleeInputs
import tensorflow as tf
import windowPositioning
import ImageAnnotator as imA


WINDOW_X = 0                                # Default image position for a window perfectly in top left corner
WINDOW_Y = 0                               # Default image position for a window perfectly in top left corner
WINDOW_WIDTH = 1280                         # Modify these values for a window snapped in top left corner
WINDOW_HEIGHT = 720  
BORDER_LEFT = 1
BORDER_RIGHT = 1
BORDER_TOP = 38
BORDER_BOTTOM = 1

def start_playing():
    inputArray = numpy.zeros(smashMeleeInputs.getSmashMeleeInputs())
    oldInputArray = inputArray
    #load the digit recognition learning
    digitAnalzer = TextAnalyzer()
    frameComparator = FrameComparator()
    

    decisionModel = tensorflowNN.create_model((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X), len(inputArray))
    screen = None
    last_time = time.time()

    while True:
        if keyboard.is_pressed("q"):
            #TODO: add dolphin termination
            cv2.destroyAllWindows()
            break



        # windowed mode
        oldScreen = screen
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))

        # takes the screen above and identifies the zone of the numbers into 6 images
        numberImages = imgProc.processNumber(screen)

        # predict a number for the 6 images
        predictions = digitAnalzer.predict(numberImages)

        score = frameComparator.compareWithLastFrame(screen,predictions)

        #add labels/annotations to the screen image for debugging purpose
        imA.addScoreToImage(screen, score)
        imA.addLabelsToImage(screen,predictions)



        cv2.imshow("window", screen) # Window showing what is captured
        cv2.waitKey(1)

        # Decision making goes here
        #predictions = decisionModel.predict(np.array([screen]))[0]
        #updateKeys(inputArray, oldInputArray, predictions)
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
    print("done")

def updateKeys(inputArray, oldInputArray, predictions):

    for index, value in np.ndenumerate(predictions):
        print(value)
        if (value > 0.95):
            inputArray[index] = 1
        else:
            inputArray[index] = 0

    for index, value in np.ndenumerate(inputArray):
        if (value != oldInputArray[index]):
            #release or press corresponding key
            if (value == 1):
                smashMeleeInputs.pressKey(index)
            elif (value == 0):
                smashMeleeInputs.releaseKey(index)
    oldInputArray = inputArray

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)


    window = windowPositioning.openWindow("Smash Melee")
    window.positionWindow(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    start_playing()

if __name__ == '__main__':
    main()