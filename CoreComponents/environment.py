
import numpy as np
import cv2
import time
import json

import smashMeleeInputs
import windowPositioning

from threading import Thread
from typing import List
from gym import spaces

from grabScreen import ScreenGraber
from FrameComparator import FrameComparator
from gameStartAnalyzer import GameStartAnalyzer
from textAnalyzer import TextAnalyzer
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT

class Environment(Thread):
    def __init__(self, env_name: str) -> None:
        self.env_name = env_name

        self.frame_comparator = FrameComparator()
        self.digit_analzer = TextAnalyzer()
        self.game_start_analyzer = GameStartAnalyzer()  
        self.screen_graber = ScreenGraber()

        if (env_name == "Smash Melee"):
            with open('../list_inputs.json', 'r') as infile:
                data = json.load(infile)
                self.action_space = spaces.Discrete(len(data))
                self.input_action_list = data
            self.inputs = smashMeleeInputs.getSmashMeleeInputs()  
            self.input_action = np.zeros(len(smashMeleeInputs.getSmashMeleeInputs()))
            self.old_input_action = self.input_action

        self.state = self.screen_graber.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
        super(Environment, self).__init__()
    
    def render(self) -> None:
        window = windowPositioning.openWindow(self.env_name)
        #print(window)
        window.positionWindow(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

    def run(self):
        def thread_function():
            last_time = time.time()
            while True:
                self.state = self.screen_graber.grab_screen_GRAY(region=(RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT))
                
                cv2.imshow("window", self.state) # Window showing what is captured
                cv2.waitKey(1)
                
                #print("screen grab took {} seconds".format(time.time()-last_time))
                last_time = time.time()

        thread = Thread(target=thread_function)
        thread.start()

    def step(self, action_index: int):

        self.input_action = self.input_action_list[action_index]
        #print(self.input_action)
        for index, input_action_value in enumerate(self.input_action):
            if (input_action_value != self.old_input_action[index]):
                #release or press corresponding key
                if (input_action_value == 1):
                    smashMeleeInputs.pressKey(index)
                elif (input_action_value == 0):
                    smashMeleeInputs.releaseKey(index)

        self.old_input_action = self.input_action

        reward = self.frame_comparator.compareWithLastFrame(self.state)

        return self.state, reward

    def wait_for_ready(self) -> None:
        if(self.env_name == "Smash Melee"):
            while True:
                cropped = self.state[130:490, 118:758]
                cropped[np.where((cropped >= 5))] = 255
                cropped = cv2.resize(cropped,(64,36))
                prediction = self.game_start_analyzer.predict(cropped)

                if prediction == 1:
                    print('GAME STARTED')
                    break

    def get_current_state(self):
        return self.state

    def stop(self):
        for index in range(len(self.input_action)):
            if(self.input_action[index] == 1):
                smashMeleeInputs.releaseKey(index)