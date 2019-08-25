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
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT, MODEL_PATH



def start_playing():
    #Create initial variables 
    screen = grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
    
    dqn_solver = DQNSolver((RECORDING_HEIGHT/2, RECORDING_WIDTH/2))
    dqn_solver.load_model(MODEL_PATH)

    
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

    screen = cv2.resize(screen, (int(RECORDING_WIDTH/2), int(RECORDING_HEIGHT/2)), interpolation=cv2.INTER_LANCZOS4)
    last_time = time.time()

    while True:
        if keyboard.is_pressed("p"):
            #TODO: add dolphin termination
            cv2.destroyAllWindows()
            dqn_solver.releaseAllKeys()
            break

        #Main decision making logic
        NN_decision_making(dqn_solver, screen)

        cv2.imshow("window", screen) # Window showing what is captured
        cv2.waitKey(1)
        
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
    print("done")

def record_sreen():
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

    while True:
        screen =  grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))

def NN_decision_making(solver, screen):
    action = solver.get_action(screen)
    solver.take_action(action)
    oldScreen = screen.copy()
    screen =  grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
    reward = frameComparator.compareWithLastFrame(screen)
    screen = cv2.resize(screen, (int(RECORDING_HEIGHT/2), int(RECORDING_WIDTH/2)), interpolation=cv2.INTER_LANCZOS4)
    solver.remember(oldScreen, action, reward, screen)
    solver.experience_replay()

def RNN_decision_making():
    print("Using RNN")

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)

    window = windowPositioning.openWindow("Smash Melee")
    print(window)
    window.positionWindow(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

    #start_playing()
    #record_screen()

if __name__ == '__main__':
    main()