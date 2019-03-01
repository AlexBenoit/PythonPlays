#!/usr/bin/python

import numpy as np
import cv2
import time
import random
import keyboard
import keyInputs
import imageProcessing as imgProc
import grabScreen
from textAnalyzer import TextAnalyzer
import tensorflowNN
import smashMeleeActions
import tensorflow as tf
from windowPositioning import openWindow, positionWindow


WINDOW_X = 7                                # Default image position for a window perfectly in top left corner
WINDOW_Y = 55                               # Default image position for a window perfectly in top left corner
WINDOW_WIDTH = 1247                         # Modify these values for a window snapped in top left corner
WINDOW_HEIGHT = 700                         # Modify these values for a window snapped in top left corner

DIGIT_WIDTH = 45
DIGIT_HEIGHT = 55

def start_playing():
    functionList = dir(smashMeleeActions)[8:]
    #load the digit recognition learning
    #digitAnalzer = TextAnalyzer()
    

    model = tensorflowNN.create_model((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X), len(functionList))
    i = 1
    screen = None
    last_time = time.time()

    while True:
        if keyboard.is_pressed("q"):
            cv2.destroyAllWindows()
            break



        # windowed mode
        oldScreen = screen
        screen =  grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
        print(screen.shape)
        # Image processing goes here if needed
        crop_img6 = screen[550:550+DIGIT_HEIGHT, 528:528+DIGIT_WIDTH]
        crop_img5 = screen[550:550+DIGIT_HEIGHT, 483:483+DIGIT_WIDTH]
        crop_img4 = screen[550:550+DIGIT_HEIGHT, 438:438+DIGIT_WIDTH]
        crop_img3 = screen[550:550+DIGIT_HEIGHT, 340:340+DIGIT_WIDTH]
        crop_img2 = screen[550:550+DIGIT_HEIGHT, 295:295+DIGIT_WIDTH]
        crop_img1 = screen[550:550+DIGIT_HEIGHT, 250:250+DIGIT_WIDTH]

        crop_img1[np.where((crop_img1 >= 5))] = 255
        crop_img6[np.where((crop_img6 >= 5))] = 255
        crop_img5[np.where((crop_img5 >= 5))] = 255
        crop_img4[np.where((crop_img4 >= 5))] = 255
        crop_img3[np.where((crop_img3 >= 5))] = 255
        crop_img2[np.where((crop_img2 >= 5))] = 255


        

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #prediction = digitAnalzer.predict(crop_img6)
        #cv2.putText(screen,str(prediction),(545,560), font, 2,(255,255,255),2,cv2.LINE_AA)
        #prediction = digitAnalzer.predict(crop_img5)
        #cv2.putText(screen,str(prediction),(500,560), font, 2,(255,255,255),2,cv2.LINE_AA)
        #prediction = digitAnalzer.predict(crop_img4)
        #cv2.putText(screen,str(prediction),(455,560), font, 2,(255,255,255),2,cv2.LINE_AA)
        #prediction = digitAnalzer.predict(crop_img3)
        #cv2.putText(screen,str(prediction),(350,560), font, 2,(255,255,255),2,cv2.LINE_AA)
        #prediction = digitAnalzer.predict(crop_img2)
        #cv2.putText(screen,str(prediction),(305,560), font, 2,(255,255,255),2,cv2.LINE_AA)
        #prediction = digitAnalzer.predict(crop_img1)
        #cv2.putText(screen,str(prediction),(260,560), font, 2,(255,255,255),2,cv2.LINE_AA)



        if keyboard.is_pressed('o'):
            print('SAVING IMAGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...')
            cv2.imwrite(r'C:\Users\Nicolas\PFE\PythonPlays\CoreComponents\assets\d'+str(i)+'.jpg', crop_img1, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            i = i+1
            cv2.imwrite(r'C:\Users\Nicolas\PFE\PythonPlays\CoreComponents\assets\d'+str(i)+'.jpg', crop_img2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            i = i+1
            cv2.imwrite(r'C:\Users\Nicolas\PFE\PythonPlays\CoreComponents\assets\d'+str(i)+'.jpg', crop_img3, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            i = i+1
            cv2.imwrite(r'C:\Users\Nicolas\PFE\PythonPlays\CoreComponents\assets\d'+str(i)+'.jpg', crop_img4, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            i = i+1
            cv2.imwrite(r'C:\Users\Nicolas\PFE\PythonPlays\CoreComponents\assets\d'+str(i)+'.jpg', crop_img5, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            i = i+1
            cv2.imwrite(r'C:\Users\Nicolas\PFE\PythonPlays\CoreComponents\assets\d'+str(i)+'.jpg', crop_img6, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            i = i+1

        
       

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
    print("starting")
    for i in range(4):
        print(i+1)
        time.sleep(1)


    openWindow()
    positionWindow()
    start_playing()

if __name__ == '__main__':
    main()