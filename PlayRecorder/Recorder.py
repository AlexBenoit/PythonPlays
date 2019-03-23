#!/usr/bin/python

#External imports
import time
import cv2
import keyboard
import keyInputs
import numpy as np
import csv
import math
from sys import getsizeof
from threading import Thread


#Internal imports
import grabScreen
import smashMeleeInputs
import windowPositioning
import Recorder

#Specific imports
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT


class Recorder(Thread):
    
    def __init__(self, index):
        Thread.__init__(self)
        np.set_printoptions(threshold=np.inf)
        self.data = []
        self.index = index
        self.input_array = np.zeros(len(smashMeleeInputs.getSmashMeleeInputs()))


    def run(self):
        print("Recording Started [" + str(self.index) + "]")
        
        keyboard.hook(self.on_press_callback)
        start = time.time()
        has_not_surpass_memory_limit = True
        data = []
        mem_size = 0
        mem_limit = 1 * math.pow(10,9) # 1 GB

        while has_not_surpass_memory_limit == True:
            screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
            mem_size = mem_size + getsizeof(screen)
            data.append([screen, self.input_array])
            if(mem_size + getsizeof(data) >  mem_limit):
                end = time.time()
                print("Recording Finished [" + str(self.index) + "] (" + str(end-start) + " seconds)")
                self.data = data
                keyboard.unhook_all()
                has_not_surpass_memory_limit = False


    def on_press_callback(self, event):
        if(event.name in keyInputs.inputDist):
            if (self.input_array[keyInputs.inputDist[event.name][0]] == 0):
                self.input_array[keyInputs.inputDist[event.name][0]] = 1
            else : 
                self.input_array[keyInputs.inputDist[event.name][0]] = 0

    def get_data(self):
        return self.data