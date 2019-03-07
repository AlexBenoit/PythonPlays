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

def create_model(input_dimension, output_length):
    width, height = input_dimension

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(width, height)),
        tf.keras.layers.Dense(LAYER1_NB_NEURONS, activation=tf.nn.relu),
        tf.keras.layers.Dense(output_length, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

class DQNSolver:

    def __init__(self, input_dimension):
        self.inputArray = np.zeros(smashMeleeInputs.getSmashMeleeInputs())
        self.oldInputArray = self.inputArray

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

        for index, input_value in np.ndenumerate(action):
            if (input_value > 0.95):
                self.inputArray[index] = 1
            else:
                self.inputArray[index] = 0
        print(self.inputArray)
        for index, old_input_value in np.ndenumerate(self.inputArray):
            if (old_input_value != self.oldInputArray[index]):
                #release or press corresponding key
                if (old_input_value == 1):
                    smashMeleeInputs.pressKey(index)
                elif (old_input_value == 0):
                    smashMeleeInputs.releaseKey(index)
        self.oldInputArray = self.inputArray

    def releaseAllKeys():
        for index in range(len(self.inputArray)):
            smashMeleeInputs.releaseKey(index)

    def experience_replay(self):
        terminal = False #DO NOT DELETE!! Needed to keep general structure 

        #if len(self.memory) < BATCH_SIZE:
        #    return

        batch = random.sample(self.memory, BATCH_SIZE)
        for oldScreen, action, reward, screen in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

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


    ##Initialize table with all zeros
    #QTable = np.zeros([env.observation_space.n, env.action_space.n])

    ##Set learning parameters
    #lr = 0.8
    #y = 0.95
    #num_episodes = 2000

    ##create lists to contain total rewards and steps per episode
    #rList = []
    #for i in range(num_episodes):
    #    #Reset environment and get first new observation
    #    state = env.reset()
    #    rAll = 0
    #    d = False
    #    j = 0

    #    #The Q-Table learning algorithm
    #    while j < 99:
    #        j += 1
    #        #Choose an action by greedily (with noise) picking from Q-Table
    #        action = np.argmax(QTable[state,:] + np.random.randn(1, env.action_space.n) * (1./(i + 1)))
    #        #Get new state and reward from environment
    #        state1, reward, d, _ = env.step(action)

    #        #Update Q-Table with new knowledge
    #        QTable[state,action] = QTable[state, action] + lr * (reward + y * np.max(QTable[state1, :]) - QTable[state, action])
    #        rAll += reward
    #        state = state1
        
    #        if d == True:
    #            break

    #    rList.append(rAll)

    #print("Score over time:", str(sum(rList)/num_episodes))
    #print("Final Q-Table values:\n", QTable)