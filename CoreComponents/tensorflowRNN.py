import numpy as np
import tensorflow as tf
import random
import gym
import os
import json

import smashMeleeInputs

from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_CERTAINTY = 0.95
BATCH_SIZE = 20

class RNNAgent(object):
    def __init__(self, nb_inputs: int, nb_outputs: int, nb_neurons=30, nb_layers=2, nb_timesteps=1, epsilon=0.9, epsilon_decay_rate=0.95, did_well_threshold=0.80):
        with open('../list_inputs.json', 'r') as infile:
            self.list_inputs = json.load(infile)
        
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.nb_neurons = nb_neurons
        self.nb_layers = nb_layers
        self.nb_timesteps = nb_timesteps

        self.batch_old_screen = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_screen = []

        self.batch_size = 20

        # Input arrays for the game
        self.inputArray = np.zeros(len(smashMeleeInputs.getSmashMeleeInputs()))
        self.oldInputArray = self.inputArray.copy()

        ### model hyperparameters
        self.epsilon = epsilon  # how much do we explore initially
        self.epsilon_decay_rate = epsilon_decay_rate  # rate by which exploration decreases, used for constant epsilon decay strategy
        self.high_score = 0  # keep track of highest score obtained thus far
        self.did_well_threshold = did_well_threshold  # how close we need to be to our high score to have "done well"
        self.network_has_had_training = False  # has our neural net had any training
        self.last_good_batch = tuple()  # memory for the last good episode we eperienced      
        self.experience = 0  # integer for keeping track of how much good experience we've had, used in custom epsilon decay function  
        self.last_100_episode_scores = deque(maxlen = 100) # keep track of average score from last 100 episodes
        
        self.sess = tf.compat.v1.InteractiveSession()
        # define the shape of the data placeholder (tensor)
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.nb_timesteps, self.nb_inputs])
        self.actions = tf.compat.v1.placeholder(tf.int32, [None])

        # define network
        #self.basic_lstm_cell = tf.keras.layers.LSTMCell(units=self.nb_neurons)
        self.learning_rate = 0.001

        self.lstm_cells = [tf.keras.layers.LSTMCell(units=self.nb_neurons) for layer in range(self.nb_layers)]
        self.multi_cell = tf.keras.layers.StackedRNNCells(self.lstm_cells)

        self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_cell, self.state, dtype=tf.float32)
        self.top_layer_h_state = self.states[-1][1]
        self.logits = tf.layers.dense(self.top_layer_h_state, self.nb_outputs)
        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)
        self.loss = tf.reduce_mean(self.xentropy, name="loss")
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)
        self.correct = tf.nn.in_top_k(self.logits, self.actions, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.sess.run(tf.compat.v1.global_variables_initializer())

    ###
    # Function to get an action from the model based on the current state of the environment.
    ###
    def get_action(self, current_state) -> int:
        #print("Predicting")
        current_state = np.array(current_state).flatten()
        current_state = np.expand_dims(current_state, axis=0)
        current_state = current_state.reshape(1, self.nb_timesteps, self.nb_inputs)
        raw_output = self.logits.eval(feed_dict={self.state: current_state})
        #print("Done predicting")
        return np.argmax(raw_output[0])

    def take_action(self, fake_action):
        action = self.list_inputs[np.argmax(fake_action)]

        for index, action_input_value in np.ndenumerate(action):
            if (action_input_value > INPUT_CERTAINTY):
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

    def remember(self, old_screen, action, reward, screen):
        self.batch_old_screen.append(np.array(old_screen).flatten().tolist())
        self.batch_action.append(np.argmax(action))
        self.batch_reward.append(reward)
        self.batch_screen.append(screen)

    def experience_replay(self):
        if len(self.batch_action) < self.batch_size:
            return

        # Get batch from total batch
        local_batch_old_screen = self.batch_old_screen[:self.batch_size]
        self.batch_old_screen = self.batch_old_screen[self.batch_size:]
        local_batch_action = self.batch_action[:self.batch_size]
        self.batch_action = self.batch_action[self.batch_size:]
        local_batch_reward = self.batch_reward[:self.batch_size]
        self.batch_reward = self.batch_reward[self.batch_size:]
        local_batch_screen = self.batch_screen[:self.batch_size]
        self.batch_screen = self.batch_screen[self.batch_size:]

        # Add total reward to last 100 batches
        episode_rewards = np.sum(local_batch_reward)
        self.last_100_episode_scores.append(episode_rewards)
        self.update_high_score(episode_rewards)

        pre_np_states = np.array(local_batch_old_screen)
        # TODO: check if 1 is nb_timesteps
        np_states = pre_np_states.reshape(pre_np_states.shape[0], 1, pre_np_states.shape[1]) # Our LSTM needs a tensor of order 3 for training
        np_actions = np.array(local_batch_action)

        batch = (np_states, np_actions)

        ## If we did well update our last good batch and amount of experience
        if self.did_we_do_well(episode_rewards):
            self.last_good_batch = batch

        self.experience += len(local_batch_old_screen)

        # Decay our epsilon using one of our two strategies 
        #wondering_gnome.decay_epsilon()
        self.decay_epsilon()

        ## Train our LSTM after every episode, but only with our most recent good batch
        self.train()

    ###
    # Function that trains the model based on the last good batch
    ###
    def train(self):
        print("Training")
        self.train_step.run(feed_dict={self.state: self.last_good_batch[0], self.actions: self.last_good_batch[1]}) # , keep_prob: 0.75})
        #print("Done training")
    
    ###
    # Function for letting us know if we did well based on the rewards received this episode and the 
    # did_well_threshold parameter.
    ###
    def did_we_do_well(self, episode_rewards):
        if episode_rewards >= self.did_well_threshold * self.high_score:
            return True
        return False

    ###
    # Function for adding an experience memory from episodes were we've "done well".
    ###
    def add_to_experience(self, episode_length):
        self.experience += episode_length

    ###
    # Function that updates our highest score acheived thus far.
    ###
    def update_high_score(self, episode_rewards):
        if episode_rewards > self.high_score:
            self.high_score = episode_rewards

    ###
    # Typical epsilon decay function.
    ###
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate

    ###
    # Customized epsilon decay function.
    ###
    def decay_epsilon_custom(self):       
        ## decaying epsilon strategy for fast solution convergence, average solution ~450 episodes
        if self.experience > 0:
            self.epsilon = 0.9
        if self.experience > 500:
            self.epsilon = 0.7
        if self.experience > 1000:
            self.epsilon = 0.6
        if self.experience > 2000:
            self.epsilon = 0.5
        if self.experience > 3000:
            self.epsilon = 0.5
        if self.experience > 4000:
            self.epsilon = 0.4
        if self.experience > 5000:
            self.epsilon = 0.3
        if self.experience > 6000:
            self.epsilon = 0.2
        if self.experience > 7000:
            self.epsilon = 0.1

    ###
    # Function for deriving epsilon from the uncertainty reflected in our LSTMs action output.
    ###
    def epsilon_from_uncertainty(self, unscaled_confidence):
            self.epsilon = 0.1  # Initialize epsilon

            ## Update epsilon if we aren't very confident
            if unscaled_confidence < 1.5: 
                confidence = unscaled_confidence / 1.5
                uncertainty = min(0.9, 1.0 - confidence)  # Uncertainty is the opposite of confidence, limit to max of 0.9
                self.epsilon = max(uncertainty, 0.1)  # Limit to min of 0.1  


if __name__ == "__main__":
    ## Data
    X_batch = np.array([
        [[0, 1, 2, 5], [9, 8, 7, 4]], # Batch 1
        [[3, 4, 5, 2], [0, 0, 0, 0]], # Batch 2
        [[6, 7, 8, 5], [6, 5, 4, 2]], # Batch 3
    ])

    env = gym.make("CartPole-v0")
    wondering_gnome = RNNAgent(4, 2, 30, 3, 1)

    for i_episode in range(10000):

        observation = env.reset()
        episode_rewards = 0
        episode_states_list = []
        episode_actions_list = []

        for t in range(200):
            env.render()

            state_for_mem = observation
            current_state = np.expand_dims(observation, axis=0)
            current_state = current_state.reshape(1,wondering_gnome.nb_timesteps,wondering_gnome.nb_inputs)
        
       
            action = env.action_space.sample()
            if wondering_gnome.network_has_had_training:
                random_fate = np.random.random()
                if random_fate > wondering_gnome.epsilon:
                    action = wondering_gnome.get_action(current_state)

        
            episode_states_list.append(observation)
            episode_actions_list.append(action)

            #Action step
            observation, reward, done, info = env.step(action)

            #add this state's reward to our episode rewards
            episode_rewards += reward

            # If we're done exit the episode loop
            if done:
                break

        ## Print some episode summary info
        print("Episode: " + str(i_episode) + ", " + "Rewards: " + str(episode_rewards))
        print("Epsilon: ")
        print(wondering_gnome.epsilon)
        print("Experince amount: ")
        print(wondering_gnome.experience)

        # Add episode score to last 100 scores
        wondering_gnome.last_100_episode_scores.append(episode_rewards)
        print("Running average of last 100 episodes: " + str(np.average(wondering_gnome.last_100_episode_scores)))

        ## If we've solved the environment close our log writer and exit
        if np.average(wondering_gnome.last_100_episode_scores) >= 195.0:
            #file_writer.close()
            print("Solved!")
            assert False

        wondering_gnome.update_high_score(episode_rewards)

        pre_np_states = np.array(episode_states_list)
        np_states = pre_np_states.reshape(pre_np_states.shape[0], 1, pre_np_states.shape[1]) # Our LSTM needs a tensor of order 3 for training
        np_actions = np.array(episode_actions_list)

        batch = (np_states, np_actions)

        ## If we did well update our last good batch and amount of experience
        if wondering_gnome.did_we_do_well(episode_rewards):

            wondering_gnome.last_good_batch = batch

            wondering_gnome.experience += len(episode_states_list)

            # Decay our epsilon using one of our two strategies 
            #wondering_gnome.decay_epsilon()
            wondering_gnome.decay_epsilon_custom()

        ## Train our LSTM after every episode, but only with our most recent good batch
        wondering_gnome.train()
    

        ## Our LSTM has been trained
        if i_episode > 1:
            wondering_gnome.network_has_had_training = True

        ## Do some logging of our current loss
        #if i_episode % 20 == 0:
            #summary_str = loss_summary.eval(feed_dict={state: wondering_gnome.last_good_batch[0], actions: wondering_gnome.last_good_batch[1]})
            #step = i_episode
            #file_writer.add_summary(summary_str, step)