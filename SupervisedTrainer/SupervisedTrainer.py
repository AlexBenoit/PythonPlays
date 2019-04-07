
#External imports
import tensorflow as tf
import numpy as np
import os, os.path
import sys
import cv2
import time

#Internal imports
import grabScreen

#Specific imports
from tensorflowNN import DQNSolver
from arrayUtility import array_to_list
from globalConstants import RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT, \
MODEL_PATH, MODEL_WEIGHTS_PATH

def main():
    # Ask start index
    index_string = input("What index should we start training ? ")
    index = int(index_string)

    files_in_directory = [name for name in os.listdir('./Training Data') if os.path.isfile("./Training Data/" + name)]

    dqn_solver = DQNSolver((RECORDING_HEIGHT/2, RECORDING_WIDTH/2))

    files_in_directory = files_in_directory[index:]

    for file in files_in_directory:
        start = time.time()
        print("Training started for : " + file)
        data = np.load('./Training Data/' + file)
        max_size_data_screens = data.f.arr_0.copy()
        resized_data_screens = []
        data_inputs = data.f.arr_1.copy()
        for i, screen in enumerate(max_size_data_screens):
            cv2.imshow("window", screen)
            cv2.waitKey(1)
            screen = cv2.resize(screen, (int((RECORDING_WIDTH)/2), int((RECORDING_HEIGHT)/2)), interpolation=cv2.INTER_LANCZOS4)
            print(screen.shape)
            resized_data_screens.append(screen)
        dqn_solver.fit(np.array(resized_data_screens), data_inputs)
        end = time.time()
        print("Time for file : " + file + " = " + str(end-start) + " seconds")

    dqn_solver.save_model(MODEL_PATH)

if __name__ == "__main__":
    main()