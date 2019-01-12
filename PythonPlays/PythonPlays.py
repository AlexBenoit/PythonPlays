#!/usr/bin/python

import numpy as np
import cv2
import time
import keyInputs
import imageProcessing as imgProc
import grabScreen

print("starting")

WINDOW_X = 1                                # Default image position for a window perfectly in top left corner
WINDOW_Y = 38                               # Default image position for a window perfectly in top left corner
WINDOW_WIDTH = 1280 - WINDOW_X * 2          # Modify these values for a window snapped in top left corner
WINDOW_HEIGHT = 720 - WINDOW_Y - WINDOW_X   # Modify these values for a window snapped in top left corner

def start_playing():
    last_time = time.time()
    while True:
        # windowed mode
        screen =  grabScreen. grab_screen_RGB(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Image processing goes here if needed

        cv2.imshow("window", screen) # Window showing what is captured

        # Decision making goes here

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

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