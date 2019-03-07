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
from tensorflowNN import DQNSolver
import smashMeleeInputs
import tensorflow as tf
import windowPositioning
import ImageAnnotator as imA
import qLearning

WINDOW_X = 0                                # Default image position for a window perfectly in top left corner
WINDOW_Y = 0                               # Default image position for a window perfectly in top left corner
WINDOW_WIDTH = 1280                         # Modify these values for a window snapped in top left corner
WINDOW_HEIGHT = 720  
BORDER_LEFT = 1
BORDER_RIGHT = 1
BORDER_TOP = 38
BORDER_BOTTOM = 1

def start_playing():
    #Create initial variables 
    screen = grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
    oldScreen = screen
    dqn_solver = DQNSolver((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X))
    
    #load the digit recognition learning
    digitAnalzer = TextAnalyzer()
    
    last_time = time.time()

    while True:
        if keyboard.is_pressed("q"):
            #TODO: add dolphin termination
            cv2.destroyAllWindows()
            dqn_solver.releaseAllKeys()
            break

        #Main decision making logic
        action = dqn_solver.get_action(screen)
        dqn_solver.take_action(action)
        oldScreen = screen
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
        #reward = compareFrames(screen, oldScreen)
        #dqn_solver.remember(oldScreen, action, reward, screen)
        #dqn_solver.experience_replay()

        # takes the screen above and identifies the zone of the numbers into 6 images
        #numberImages = imgProc.processNumber(screen)

        # predict a number for the 6 images
        #predictions = digitAnalzer.predict(numberImages)

        #add labels/annotations to the screen image for debugging purpose
        #imA.addLabelsToImage(screen,predictions)



        cv2.imshow("window", screen) # Window showing what is captured
        cv2.waitKey(1)
        
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
    print("done")

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)


    window = windowPositioning.openWindow("Smash Melee")
    window.positionWindow(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    print("Go in the game")
    timeLeft = 45
    for i in range(timeLeft):
        print(timeLeft - i)
        time.sleep(1)

    start_playing()

if __name__ == '__main__':
    main()