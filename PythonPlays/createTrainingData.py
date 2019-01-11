import numpy as np
import grabScreen
import cv2
import time
import logKeyInputs as logKeys
import os

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

def keys_to_output(keys):
    output = [0,0,0]

    if "A" in keys:
        output[0] = 1
    elif "D" in keys:
        output[2] = 1
    else:
        output[1] = 1

    return output

def create_training_data():
    last_time = time.time()
    while True:
        # windowed mode
        screen =  grabScreen.grab_screen(region=(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
        screen = cv2.cvtColor(screate_training_data, cv2.COLOR_RGB2GRAY)
        screen = cv2.resize(screen, (80,60))
        keys = logKeys.key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

    print("done")

def main():

    file_name = 'training_data.npy'

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    for i in range(4):
        print(i+1)
        time.sleep(1)

    create_training_data()

if __name__ == '__main__':
    main()