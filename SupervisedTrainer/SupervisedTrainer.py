
#External imports
import numpy as np

#Internal imports
import grabScreen

#Specific imports
from tensorflowNN import DQNSolver
from globalConstants import WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT, BORDER_LEFT, \
BORDER_RIGHT, BORDER_TOP, BORDER_BOTTOM, MODEL_PATH, MODEL_WEIGHTS_PATH


#import data
screen_data = grabScreen.grab_screen_GRAY(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)) #TODO: update to data from import
input_data = np.zeros(18) #TODO: update to data from import

dqn_solver = DQNSolver((WINDOW_HEIGHT - WINDOW_Y, WINDOW_WIDTH - WINDOW_X))

dqn_solver.fit(np.array([screen_data]), np.array([input_data]))
dqn_solver.save_weights(MODEL_WEIGHTS_PATH)
dqn_solver.save_model(MODEL_PATH)