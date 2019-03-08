#!/usr/bin/python

#External imports
import time
import cv2
import keyboard
import numpy as np

#Internal imports
import grabScreen
import smashMeleeInputs

#Specific imports
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT

input_array = np.zeros(len(smashMeleeInputs.getSmashMeleeInputs()))

def on_press_callback(event):
    print("pressed key")
    print(event.name)

def on_release_callback(event):
    print("released key")
    print(event.name)

def start_recording():
    ready_to_record = False

    while not ready_to_record:
        if keyboard.is_pressed("r"):
            ready_to_record = True

    keyboard.on_press(on_press_callback)

    while True:
        if keyboard.is_pressed("q"):
            break

        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))

        #cv2.imshow("window", screen) # Window showing what is captured
        #cv2.waitKey(1)

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        #time.sleep(1)

    #window = windowPositioning.openWindow("Smash Melee")
    #window.positionWindow(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    start_recording()

if __name__ == '__main__':
    main()