#!/usr/bin/python

#External imports
import numpy as np
import cv2
import time
import random
import keyboard
import tensorflow as tf
import json

#Internal imports
import keyInputs
import imageProcessing as imgProc
import grabScreen

import smashMeleeInputs
import windowPositioning
import ImageAnnotator as imA


#Specific imports
from environment import Environment
from FrameComparator import FrameComparator
from gameStartAnalyzer import GameStartAnalyzer
from textAnalyzer import TextAnalyzer
from model import Model
from tensorflowNN import DQNSolver
from tensorflowRNN import RNNAgent
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT, MODEL_PATH

def start_playing():
    # Load the digit recognition learning
    frameComparator = FrameComparator()
    digitAnalzer = TextAnalyzer()
    gameStartAnalyzer = GameStartAnalyzer()

    #Create initial variables 
    screen = grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
    list_inputs = None
    with open('../list_inputs.json', 'r') as infile:
        list_inputs = json.load(infile)

    dqn_solver = RNNAgent(RECORDING_HEIGHT*RECORDING_WIDTH, len(list_inputs))
    #dqn_solver = DQNSolver((RECORDING_HEIGHT/2, RECORDING_WIDTH/2))
    #dqn_solver.load_model(MODEL_PATH)

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

    #screen = cv2.resize(screen, (int(RECORDING_WIDTH/2), int(RECORDING_HEIGHT/2)), interpolation=cv2.INTER_LANCZOS4)
    last_time = time.time()

    while True:
        if keyboard.is_pressed("p"):
            #TODO: add dolphin termination
            cv2.destroyAllWindows()
            dqn_solver.releaseAllKeys()
            break

        #Main decision making logic
        screen = RNN_decision_making(dqn_solver, screen, frameComparator)
        
        cv2.imshow("window", screen) # Window showing what is captured
        cv2.waitKey(1)
        
        #print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
    #print("done")

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

    return screen

def RNN_decision_making(solver, screen, frameComparator):
    #action = np.random.choice(list_inputs) # Initialize action as a random possible action
    #random_fate = np.random.random()
    #if random_fate > wondering_gnome.epsilon:
    action = solver.get_action(screen)
    solver.take_action(action)
    oldScreen = screen.copy()
    screen = grabScreen.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
    reward = frameComparator.compareWithLastFrame(screen)
    solver.remember(oldScreen, action, reward, screen)
    solver.experience_replay()

    return screen

def new_playing():
    
    env = Environment("Smash Melee")
    agent = Model((int(RECORDING_HEIGHT/4), int(RECORDING_WIDTH/4), 1), (env.action_space.n,))
    agent.load_model(MODEL_PATH)
    #env.render()
    env.start()
    #env.wait_for_ready()
    
    while True:
        state = env.get_current_state()
        screen = cv2.resize(state, (int(RECORDING_HEIGHT/4), int(RECORDING_WIDTH/4)))
        action = np.argmax(agent.predict(np.array([screen]))[0])
        print(action)
        new_state, reward = env.step(action)

        #agent.remember(state, action, reward, new_state)
        #agent.experience_replay()

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)

    new_playing()

if __name__ == '__main__':
    main()