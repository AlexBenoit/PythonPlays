
#External imports
import numpy as np
import os, os.path
import sys
import cv2
import time

#Internal imports
import grabScreen

#Specific imports
from tensorflowNN import DQNSolver
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, \
MODEL_PATH, MODEL_WEIGHTS_PATH

def main():
    # Ask start index
    index_string = input("What index should we start training ? ")
    index = int(index_string)

    files_in_directory = [name for name in os.listdir('./Training Data') if os.path.isfile("./Training Data/" + name)]

    dqn_solver = DQNSolver((360,640))

    files_in_directory = files_in_directory[index:]

    for file in files_in_directory:
        start = time.time()
        print("Training started for : " + file)
        data = np.load('./Training Data/' + file)
        main_array = data.f.arr_0
        max_res_input_array = np.array(array_to_list(main_array[::10,0], 1))
        resize_input_array = []
        for i, screen in enumerate(max_res_input_array):
            width = int(screen.shape[1] * 50/ 100)
            height = int(screen.shape[0] * 50/ 100)
            dim = (width, height)
            resize_input_array.append(cv2.resize(screen, dim, interpolation=cv2.INTER_LANCZOS4))
        output_array = np.array(array_to_list(main_array[::10,1], 1))
        dqn_solver.fit(np.array(resize_input_array), output_array)
        end = time.time()
        print("Time for file : " + file + " = " + str(end-start) + " seconds")


    dqn_solver.save_weights(MODEL_WEIGHTS_PATH)
    dqn_solver.save_model(MODEL_PATH)

def array_to_list(array, level):
    print("converting to list")
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