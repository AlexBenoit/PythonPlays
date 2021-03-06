import tensorflow as tf
import numpy as np

from collections import deque

GAMMA = 0.95
LEARNING_RATE = 0.01
MEMORY_SIZE = 1000000
BATCH_SIZE = 20

class Model(object):
    """description of class"""
    def __init__(self, input_shape, output_shape, timesteps=20, type="CNN"):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.timesteps = timesteps
        self.type = type

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.last_100_episode_scores = deque(maxlen = BATCH_SIZE) # keep track of average score from last 100 episodes
        self.batch_size = BATCH_SIZE
        self.high_score = 0  # keep track of highest score obtained thus far
        self.experience = 0  # integer for keeping track of how much good experience we've had, used in custom epsilon decay function
        self.did_well_threshold = 0.80  # how close we need to be to our high score to have "done well"
        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_new_state = []
        # Model with a conv2d layer in a RNN
        if self.type == "CRNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (5, 5), input_shape = (None, timesteps) + input_shape, activation = 'relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(5,5))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.LSTM(units=64, activation=tf.nn.relu),
                tf.keras.layers.Dense(69, activation=tf.nn.softmax)
            ])

            self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        # Model of a RNN
        elif self.type == "RNN":
            ### model hyperparameters
            self.epsilon = 0.9  # how much do we explore initially
            self.epsilon_decay_rate = 0.95  # rate by which exploration decreases, used for constant epsilon decay strategy
            self.high_score = 0  # keep track of highest score obtained thus far
            self.did_well_threshold = 0.8  # how close we need to be to our high score to have "done well"
            self.network_has_had_training = False  # has our neural net had any training
            self.last_good_batch = tuple()  # memory for the last good episode we eperienced      
            self.experience = 0  # integer for keeping track of how much good experience we've had, used in custom epsilon decay function  
            self.last_100_episode_scores = deque(maxlen = 100) # keep track of average score from last 100 episodes
        
            self.sess = tf.compat.v1.InteractiveSession()
            # define the shape of the data placeholder (tensor)
            self.state = tf.compat.v1.placeholder(tf.float32, [None, 180*219, 1])
            self.actions = tf.compat.v1.placeholder(tf.int32, [None])

            # define network
            #self.basic_lstm_cell = tf.keras.layers.LSTMCell(units=self.nb_neurons)
            self.learning_rate = 0.001

            self.lstm_cells = [tf.keras.layers.LSTMCell(units=300) for layer in range(1)]
            self.multi_cell = tf.keras.layers.StackedRNNCells(self.lstm_cells)

            self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_cell, self.state, dtype=tf.float32)
            self.top_layer_h_state = self.states[-1][1]
            self.logits = tf.layers.dense(self.top_layer_h_state, 69)
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)
            self.loss = tf.reduce_mean(self.xentropy, name="loss")
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)
            self.correct = tf.nn.in_top_k(self.logits, self.actions, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

            self.sess.run(tf.compat.v1.global_variables_initializer())

        elif self.type == "CNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (5, 5), input_shape = input_shape, activation = 'relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(5,5)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.output_shape[0], activation=tf.nn.softmax)
            ])

            self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, X, Y):
        print("Fitting model")
        if self.type == "CRNN":
            new_X = X.reshape((-1, self.timesteps) + self.input_shape).copy()
            new_Y = Y.reshape((-1, self.timesteps)).copy()
            self.model.fit(new_X, new_Y)

        elif self.type == "CNN":
            new_X = X.reshape((-1,) + self.input_shape).copy()
            self.model.fit(new_X, Y)

        elif self.type == "RNN":
            self.train_step.run(feed_dict={self.state: X, self.actions: Y})

    def predict(self, X):
        new_X = X.reshape((-1,) + self.input_shape).copy()
        return self.model.predict(new_X)

    def remember(self, state, action, reward, new_state):
        self.batch_state.append(state)
        self.batch_action.append(action)
        self.batch_reward.append(reward)
        self.batch_new_state.append(new_state)

    def experience_replay(self):
        if self.type == "CNN":
            if len(self.batch_action) < BATCH_SIZE:
                return

            print("Experiencing")
            screens = []
            inputs = []

            # Get batch from total batch
            local_batch_state = self.batch_state[:self.batch_size]
            self.batch_state = self.batch_state[self.batch_size:]
            local_batch_action = self.batch_action[:self.batch_size]
            self.batch_action = self.batch_action[self.batch_size:]
            local_batch_reward = self.batch_reward[:self.batch_size]
            self.batch_reward = self.batch_reward[self.batch_size:]
            local_batch_new_state = self.batch_new_state[:self.batch_size]
            self.batch_new_state = self.batch_new_state[self.batch_size:]

            # Add total reward to last 100 batches
            episode_rewards = np.sum(local_batch_reward)
            self.last_100_episode_scores.append(episode_rewards)
            self.update_high_score(episode_rewards)

            batch = (np.array(local_batch_state), np.array(local_batch_action))

            ## If we did well update our last good batch and amount of experience
            if self.did_we_do_well(episode_rewards):
                self.last_good_batch = batch

            self.experience += len(local_batch_state)

            self.train()

    ###
    # Function that updates our highest score acheived thus far.
    ###
    def update_high_score(self, episode_rewards):
        if episode_rewards > self.high_score:
            self.high_score = episode_rewards
        print("HIGH SCORE: " + str(self.high_score))

    ###
    # Function for letting us know if we did well based on the rewards received this episode and the 
    # did_well_threshold parameter.
    ###
    def did_we_do_well(self, episode_rewards):
        if episode_rewards >= self.did_well_threshold * self.high_score:
            return True
        return False

    def train(self):
        print("Training model based on last good batch")
        self.fit(self.last_good_batch[0], self.last_good_batch[1])

    def save_model(self, path):
        self.model.save(path + "model_" + self.type + ".h5")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path + "model_" + self.type + ".h5")