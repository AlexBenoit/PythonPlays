import tensorflow as tf
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import smashMeleeInputs

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.995

LAYER1_NB_NEURONS = 128

class DQNSolver:

    def __init__(self, input_dimension):
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
            tf.keras.layers.Flatten(input_shape=(width, height)),
            tf.keras.layers.Dense(LAYER1_NB_NEURONS, activation=tf.nn.relu),
            tf.keras.layers.Dense(len(self.inputArray), activation=tf.nn.sigmoid)
        ])

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def remember(self, oldScreen, action, reward, screen):
        self.memory.append((oldScreen, action, reward, screen))

    def get_action(self, screen):
        #image processing

        #if np.random.rand() < self.exploration_rate:
            #return random.randrange(self.action_space)
        q_values = self.model.predict(np.array([screen]))
        return q_values[0]

    def take_action(self, action):

        for index, action_input_value in np.ndenumerate(action):
            if (action_input_value > 0.95):
                self.inputArray[index] = 1
            else:
                self.inputArray[index] = 0
        #print(self.inputArray)
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
        terminal = False #DO NOT DELETE!! Needed to keep general structure 

        if len(self.memory) < BATCH_SIZE:
            return

        def updateQValue(value):
            return reward + GAMMA * value

        batch = random.sample(self.memory, BATCH_SIZE)
        for oldScreen, action, reward, screen in batch:
            q_update = self.model.predict(np.array([screen]))[0]
            if not terminal:
                #q_update = (reward + GAMMA * np.amax(self.model.predict(np.array([screen]))[0]))
                q_update = np.apply_along_axis(updateQValue, 0, self.model.predict(np.array([screen]))[0])
            #q_values = self.model.predict(state)
            #q_values[0][action] = q_update
            self.model.fit(np.array([oldScreen]), np.array([q_update]), verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def fit(self, input_data, output_data):
        print("Fitting model")
        self.model.fit(input_data, output_data)

    def save_weights(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def load_model(self, path):
        self.model.load_model(path)

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