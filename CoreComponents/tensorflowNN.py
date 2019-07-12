#External imports
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym
import random
import random
import time
import json

#Internal imports
import smashMeleeInputs

#Specific imports
from collections import deque
from threading import Thread
from arrayUtility import array_to_list
from globalConstants import RECORDING_WIDTH, RECORDING_HEIGHT, MODEL_PATH

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001
INPUT_CERTAINTY = 0.95
MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.995

LAYER1_NB_NEURONS = 128

class DQNSolver:

    def __init__(self, input_dimension):
        with open('../list_inputs.json', 'r') as infile:
            self.list_inputs = json.load(infile)
            self.inputArray = np.zeros(len(smashMeleeInputs.getSmashMeleeInputs()))
            self.oldInputArray = self.inputArray.copy()
            self.exploration_rate = EXPLORATION_MAX

            #self.action_space = action_space
            self.memory = deque(maxlen=MEMORY_SIZE)

            # MODEL FOR CARTPOLE
            #self.model = tf.keras.sequential()
            #self.model.add(tf.keras.layers.dense(24, input_shape=(observation_space,), activation="relu"))
            #self.model.add(tf.keras.layers.dense(24, activation="relu"))
            #self.model.add(tf.keras.layers.dense(action_space, activation="linear"))
            #self.model.compile(loss="mse", optimizer=tf.keras.optimizers.adam(lr=learning_rate))

            width, height = input_dimension

            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), input_shape = (width, height, 1), activation = 'relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=LAYER1_NB_NEURONS, activation=tf.nn.relu),
                tf.keras.layers.Dense(len(self.list_inputs), activation=tf.nn.softmax)
            ])

            self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    def remember(self, oldScreen, action, reward, screen):
        self.memory.append((oldScreen, action, reward, screen))

    def get_action(self, screen):
        #image processing

        #if np.random.rand() < self.exploration_rate:
            #return random.randrange(self.action_space)
        screen = np.reshape(screen, (int(RECORDING_HEIGHT/2), int(RECORDING_WIDTH/2), 1))
        q_values = self.model.predict(np.array([screen]))
        return q_values[0]

    def take_action(self, fake_action):
        action = self.list_inputs[np.argmax(fake_action)]

        for index, action_input_value in np.ndenumerate(action):
            if (action_input_value > INPUT_CERTAINTY):
                self.inputArray[index] = 1
            else:
                self.inputArray[index] = 0
        print(self.inputArray)
        for index, input_value in np.ndenumerate(self.inputArray):
            if (input_value != self.oldInputArray[index[0]]):
                #release or press corresponding key
                if (input_value == 1):
                    smashMeleeInputs.pressKey(index[0])
                elif (input_value == 0):
                    smashMeleeInputs.releaseKey(index[0])
        self.oldInputArray = self.inputArray

    def releaseAllKeys(self):
        for index in range(len(self.inputArray)):
            if(self.inputArray[index] == 1):
                smashMeleeInputs.releaseKey(index)

    def experience_replay(self): 
        if len(self.memory) < BATCH_SIZE:
            return

        #learner = Learner(self)
        #learner.start()

        terminal = False #DO NOT DELETE!! Needed to keep general structure
        
        data_screens = []
        data_inputs = []
        batch = random.sample(self.memory, BATCH_SIZE)
        for oldScreen, action, reward, screen in batch:
            if not terminal:
                screen = np.reshape(screen, (int(RECORDING_HEIGHT/2), int(RECORDING_WIDTH/2), 1))
                q_update = self.model.predict(np.array([screen]))[0]
                for index_2, (value) in enumerate(np.nditer(q_update, op_flags=['readwrite'])):
                    value[...] = (1 - LEARNING_RATE) * action[index_2] + LEARNING_RATE * (reward + GAMMA * value)
            oldScreen = np.reshape(oldScreen, (int(RECORDING_HEIGHT/2), int(RECORDING_WIDTH/2), 1))
            data_screens.append(oldScreen)
            data_inputs.append(q_update)
        
        self.model.fit(np.array(data_screens), np.array(data_inputs))
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        try:
            self.save_model(MODEL_PATH)
        except:
            print("COULD NOT SAVE MODEL")
        

    def fit(self, input_data, output_data):
        print("Fitting model")
        real_output_data = []
        for data in output_data:
            data_append = np.zeros(len(self.list_inputs))
            for index, input in enumerate(self.list_inputs):
                if np.array_equal(input, data):
                    data_append[index] = 1
                    print(data_append)
                    real_output_data.append(data_append)
        real_output_data = np.array(real_output_data)
        self.model.fit(input_data, np.array(real_output_data))

    def save_weights(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

class Learner(Thread):
    def __init__(self,solver):
        Thread.__init__(self)
        self.solver = solver
 
    def run(self):
        terminal = False #DO NOT DELETE!! Needed to keep general structure
        
        data_screens = []
        data_inputs = []
        batch = random.sample(self.solver.memory, BATCH_SIZE)
        for oldScreen, action, reward, screen in batch:
            if not terminal:
                screen = np.reshape(screen, (int(RECORDING_HEIGHT/2), int(RECORDING_WIDTH/2), 1))
                q_update = self.solver.model.predict(np.array([screen]))[0]
                for index_2, (value) in enumerate(np.nditer(q_update, op_flags=['readwrite'])):
                    value[...] = (1 - LEARNING_RATE) * action[index_2] + LEARNING_RATE * (reward + GAMMA * value)
            oldScreen = np.reshape(oldScreen, (int(RECORDING_HEIGHT/2), int(RECORDING_WIDTH/2), 1))
            data_screens.append(oldScreen)
            data_inputs.append(q_update)
        
        self.solver.model.fit(np.array(data_screens), np.array(data_inputs))
        self.solver.exploration_rate *= EXPLORATION_DECAY
        self.solver.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        try:
            self.solver.save_model(MODEL_PATH)
        except:
            print("COULD NOT SAVE MODEL")

if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0

        while True:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run:", str(run), ", exploration:", str(dqn_solver.exploration_rate), ", score:", str(step))
                break 
            dqn_solver.experience_replay()