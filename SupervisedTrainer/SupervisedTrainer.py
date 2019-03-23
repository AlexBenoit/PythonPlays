
#External imports
import numpy as np
import os, os.path
import sys


#Internal imports
import grabScreen

#Specific imports
from tensorflowNN import DQNSolver
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, BORDER_LEFT, \
BORDER_RIGHT, BORDER_TOP, BORDER_BOTTOM, MODEL_PATH, MODEL_WEIGHTS_PATH

# Ask start index
index_string = input("What index should we start training ? ")
index = int(index_string)


files_in_directory = [name for name in os.listdir('./input') if os.path.isfile(name)]

dqn_solver = DQNSolver((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X))


for index_file in range(0, index):
    del files_in_directory[index_file]

for file in files_in_directory:
    print("Training started for : " + file)
    data = np.load('./input/' + file)
    for captured_data in data:
        screen_data = captured_data[0]
        input_data = captured_data[1]
        dqn_solver.fit(np.array([screen_data]), np.array([input_data]))


dqn_solver.save_weights(MODEL_WEIGHTS_PATH)
dqn_solver.save_model(MODEL_PATH)