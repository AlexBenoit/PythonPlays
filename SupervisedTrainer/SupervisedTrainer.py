
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
from globalConstants import RECORDING_X, RECORDING_Y, RECORDING_WIDTH, RECORDING_HEIGHT, \
MODEL_PATH, MODEL_WEIGHTS_PATH

def main():
    # Ask start index
    index_string = input("What index should we start training ? ")
    index = int(index_string)

    files_in_directory = [name for name in os.listdir('./Training Data') if os.path.isfile("./Training Data/" + name)]

    dqn_solver = DQNSolver(((RECORDING_HEIGHT - RECORDING_Y)/2, (RECORDING_WIDTH - RECORDING_X)/2))

    files_in_directory = files_in_directory[index:]

    for file in files_in_directory:
        start = time.time()
        print("Training started for : " + file)
        data = np.load('./Training Data/' + file)
        max_size_data_screens = data.f.arr_0[::10].copy()
        resized_data_screens = []
        data_inputs = data.f.arr_1[::10].copy()
        for i, screen in enumerate(max_size_data_screens):
            print(data_inputs[i])
            cv2.imshow("window", screen)
            cv2.waitKey(5)
            screen = cv2.resize(screen, (int((RECORDING_WIDTH - RECORDING_X)/2), int((RECORDING_HEIGHT - RECORDING_Y)/2)), interpolation=cv2.INTER_LANCZOS4)
            resized_data_screens.append(screen)
        dqn_solver.fit(np.array(resized_data_screens), data_inputs)
        end = time.time()
        print("Time for file : " + file + " = " + str(end-start) + " seconds")

    dqn_solver.save_model(MODEL_PATH)

def array_to_list(array, level):
    if level == 0:
        return array

    if isinstance(array, np.ndarray):
        return array_to_list(array.tolist(), level - 1)
    elif isinstance(array, list):
        return [array_to_list(item, level - 1) for item in array]
    elif isinstance(array, tuple):
        return tuple(array_to_list(item, level - 1) for item in array)
    else:
        return array

if __name__ == "__main__":
    main()