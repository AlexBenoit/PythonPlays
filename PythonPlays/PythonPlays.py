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
import smashMeleeInputs
import windowPositioning
import ImageAnnotator as imA
from gameStartAnalyzer import GameStartAnalyzer

#Specific imports
from textAnalyzer import TextAnalyzer
from tensorflowNN import DQNSolver
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT



def start_playing():
    #Create initial variables 
    screen = grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
    dqn_solver = DQNSolver((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X))
    
    #load the digit recognition learning
    digitAnalzer = TextAnalyzer()
    gameStartAnalyzer = GameStartAnalyzer()

    # loop until game start
    while True:
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
        cropped = screen[130:490, 320:960]
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
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
        #reward = compareFrames(screen, oldScreen)
        reward = 1
        dqn_solver.remember(oldScreen, action, reward, screen)
        dqn_solver.experience_replay()

        #cv2.imshow("window", screen) # Window showing what is captured
        #cv2.waitKey(1)
        
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


    start_playing()

if __name__ == '__main__':
    main()