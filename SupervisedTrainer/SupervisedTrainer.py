
#External imports
import numpy as np
import os, os.path
import sys

#Internal imports
import grabScreen

#Specific imports
#from tensorflowNN import DQNSolver
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, \
MODEL_PATH, MODEL_WEIGHTS_PATH

def main():
    # Ask start index
    index_string = input("What index should we start training ? ")
    index = int(index_string)

    files_in_directory = [name for name in os.listdir('./Training Data') if os.path.isfile("./Training Data/" + name)]

    dqn_solver = DQNSolver((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X))

    files_in_directory = files_in_directory[index:]

    for file in files_in_directory:
        print("Training started for : " + file)
        data = np.load('./Training Data/' + file)
        main_array = data.f.arr_0
        dqn_solver.fit(np.array(array_to_list(main_array[::10,0], 1)), np.array(array_to_list(main_array[::10,1], 1)))


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
    #main()

    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    with tf.device('/cpu:0'):
        print("using gpu")