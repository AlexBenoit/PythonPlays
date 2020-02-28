#!/usr/bin/python

#External imports
import numpy as np
import cv2
import time
import random
import keyboard
import tensorflow as tf
import json
import os
import psutil

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

def kill_proc_tree(pid, including_parent=True):    
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()

def start_playing():
    
    env = Environment("Smash Melee")
    agent = Model((int(RECORDING_HEIGHT/4), int(RECORDING_WIDTH/4), 1), (env.action_space.n,))
    agent.load_model(MODEL_PATH)
    env.render()
    env.start()
    env.wait_for_ready()
    
    while True:
        if keyboard.is_pressed("p"):
            print("TERMINATED")
            #TODO: add dolphin termination
            cv2.destroyAllWindows()
            env.stop()
            agent.save_model(MODEL_PATH)
            me = os.getpid()
            kill_proc_tree(me)
            break

        state = env.get_current_state()
        state = cv2.resize(state, (int(RECORDING_WIDTH/4), int(RECORDING_HEIGHT/4)))
        action = np.argmax(agent.predict(np.array([state]))[0])
        new_state, reward = env.step(action)
        new_state = cv2.resize(new_state, (int(RECORDING_WIDTH/4), int(RECORDING_HEIGHT/4)))
        #agent.remember(state, action, reward, new_state)
        #agent.experience_replay()

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)

    start_playing()

if __name__ == '__main__':
    main()