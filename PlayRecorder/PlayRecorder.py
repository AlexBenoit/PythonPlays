#!/usr/bin/python

#External imports
import time
import cv2
import keyboard
import keyInputs
import numpy as np
import csv


#Internal imports
import grabScreen
import smashMeleeInputs

#Specific imports
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT

input_array = np.zeros(len(smashMeleeInputs.getSmashMeleeInputs()))

recording = False

def on_press_callback(event):
   if(event.name in keyInputs.inputDist):
       if (input_array[keyInputs.inputDist[event.name][0]] == 0):
           input_array[keyInputs.inputDist[event.name][0]] = 1
       else : 
           input_array[keyInputs.inputDist[event.name][0]] = 0

def start_recording():
    ready_to_record = False

    keyboard.hook(on_press_callback)

    while not ready_to_record:
        if keyboard.is_pressed("r"):
            ready_to_record = True

    recording = True
    f = open('data.csv', 'w')

    while recording == True:
        if  keyboard.is_pressed("esc"):
           recording = False
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
        f.write(str(screen))
        f.write('\n')
        print(recording)
    keyboard.unhook_all()
    f.close()
    

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        #time.sleep(1)

    #window = windowPositioning.openWindow("Smash Melee")
    #window.positionWindow(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    np.set_printoptions(threshold=np.inf)

    start_recording()

if __name__ == '__main__':
    main()