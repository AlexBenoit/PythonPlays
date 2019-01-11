#!/usr/bin/python

import numpy as np
import cv2
import time
import keyInputs
import imageProcessing as imgProc
import grabScreen

print("starting")

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

def start_playing():
    last_time = time.time()
    while True:
        # windowed mode
        screen =  grabScreen. grab_screen_RGB(region=(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
        #new_screen, original_image, m1, m2 = imgProc.process_img(screen)

        #cv2.imshow("window", screen)
        #cv2.imshow("window2", screen)

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