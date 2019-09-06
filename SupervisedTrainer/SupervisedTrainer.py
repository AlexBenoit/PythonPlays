
#External imports
import numpy as np
import os, os.path
import sys
import cv2
import time
import json
import tensorflow as tf
#Internal imports
import grabScreen

#Specific imports
from tensorflowNN import DQNSolver
from model import Model
from arrayUtility import array_to_list
from globalConstants import RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT, \
MODEL_PATH, MODEL_WEIGHTS_PATH

def sorting_function(text):
    number = int(''.join(filter(str.isdigit, text)))
    return number

def main():
    list_inputs = None
    with open('../list_inputs.json', 'r') as infile:
            list_inputs = json.load(infile)
    files_in_directory = [name for name in os.listdir('./Training Data') if os.path.isfile("./Training Data/" + name)]
    files_in_directory.sort(key=sorting_function)
    files_in_directory = files_in_directory[:]
    dqn_solver = Model((int(RECORDING_HEIGHT/4), int(RECORDING_WIDTH/4), 1), (69,))

    for file in files_in_directory:
        start = time.time()
        print("Training started for : " + file)
        try:
            data = np.load('./Training Data/' + file)
        except:
            print("Can not load file")
        screens = data.f.arr_0[:].copy()
        inputs = data.f.arr_1[:].copy()
        screens_resized = []
        output_data = []
        for i, screen in enumerate(screens):
            cv2.imshow("window", screen) # Window showing what is captured
            cv2.waitKey(1)
            screen = cv2.resize(screen, (int((RECORDING_WIDTH)/4), int((RECORDING_HEIGHT)/4)))
            screens_resized.append(screen)
            for index, input in enumerate(list_inputs):
                if np.array_equal(input, inputs[i]):
                    #value = np.zeros(len(list_inputs))
                    #value[index] = 1
                    output_data.append(index)

        dqn_solver.fit(np.array(screens_resized, dtype=np.float32), np.array(output_data))
        end = time.time()
        print("Time for file : " + file + " = " + str(end-start) + " seconds")       
        dqn_solver.save_model(MODEL_PATH)

if __name__ == "__main__":
    main()