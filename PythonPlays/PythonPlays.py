#!/usr/bin/python

import numpy as np
import cv2
import time
import random
import keyboard
import keyInputs
import imageProcessing as imgProc
import grabScreen
import tensorflowNN
import smashMeleeActions
import tensorflow as tf

print("starting")

WINDOW_X = 1                                # Default image position for a window perfectly in top left corner
WINDOW_Y = 38                               # Default image position for a window perfectly in top left corner
WINDOW_WIDTH = 1280                         # Modify these values for a window snapped in top left corner
WINDOW_HEIGHT = 720                         # Modify these values for a window snapped in top left corner

def start_playing():
    functionList = dir(smashMeleeActions)[8:]
    model = tensorflowNN.create_model((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X), len(functionList))
    screen = None
    last_time = time.time()

    while True:
        if keyboard.is_pressed("q"):
            cv2.destroyAllWindows()
            break

        # windowed mode
        oldScreen = screen
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))

        # Image processing goes here if needed

        cv2.imshow("window", screen) # Window showing what is captured
        cv2.waitKey(1)

        # Decision making goes here
        predictions = model.predict(np.array([screen]))
        try:
            getattr(smashMeleeActions, functionList[random.randint(0, len(functionList) - 1)])()
        except:
            print("Action failed")

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
    print("done")

def main():

    for i in range(4):
        print(i+1)
        time.sleep(1)

    start_playing()

if __name__ == '__main__':
    main()