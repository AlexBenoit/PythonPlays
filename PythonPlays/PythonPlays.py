#!/usr/bin/python

#External imports
import numpy as np
import cv2
import time
import random
import keyboard
import tensorflow as tf

#Internal imports
import keyInputs
import imageProcessing as imgProc
import grabScreen
from FrameComparator import FrameComparator
import smashMeleeInputs
import windowPositioning
import ImageAnnotator as imA
from gameStartAnalyzer import GameStartAnalyzer

#Specific imports
from textAnalyzer import TextAnalyzer
from tensorflowNN import DQNSolver
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT



def start_playing():
    #Create initial variables 
    screen = grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
    dqn_solver = DQNSolver((RECORDING_HEIGHT/2, RECORDING_WIDTH/2))
    
    #load the digit recognition learning
    frameComparator = FrameComparator()
    digitAnalzer = TextAnalyzer()
    gameStartAnalyzer = GameStartAnalyzer()

    # loop until game start
    while True:
        screen =  grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
        cropped = screen[130:490, 118:758]
        cropped[np.where((cropped >= 5))] = 255
        cropped = cv2.resize(cropped,(64,36))
        prediction = gameStartAnalyzer.predict(cropped)

        if prediction == 1:
            print('GAME STARTED')
            break

    
    last_time = time.time()

    while True:
        if keyboard.is_pressed("p"):
            #TODO: add dolphin termination
            cv2.destroyAllWindows()
            dqn_solver.releaseAllKeys()
            break

        #Main decision making logic
        action = dqn_solver.get_action(screen)
        dqn_solver.take_action(action)
        oldScreen = screen.copy()
        screen =  grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
        #reward = frameComparator.compareWithLastFrame(screen)
        #dqn_solver.remember(oldScreen, action, reward, screen)
        #dqn_solver.experience_replay()

        cv2.imshow("window", screen) # Window showing what is captured
        cv2.waitKey(1)
        
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
    print("done")

def main():
    window = windowPositioning.openWindow("Smash Melee")
    window.positionWindow(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

    start_playing()

if __name__ == '__main__':
    main()