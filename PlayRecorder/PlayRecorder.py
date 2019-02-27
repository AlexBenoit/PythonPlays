#!/usr/bin/python

import time
import cv2
import grabScreen

WINDOW_X = 1                                # Default image position for a window perfectly in top left corner
WINDOW_Y = 38                               # Default image position for a window perfectly in top left corner
WINDOW_WIDTH = 1280                         # Modify these values for a window snapped in top left corner
WINDOW_HEIGHT = 720                         # Modify these values for a window snapped in top left corner

def start_recording():
    screen = None

    while True:
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))

        cv2.imshow("window", screen) # Window showing what is captured
        cv2.waitKey(1)

def main():
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)

    start_recording()

if __name__ == '__main__':
    main()