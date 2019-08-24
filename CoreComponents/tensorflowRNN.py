import numpy as np
import tensorflow as tf
import random
import gym

from collections import deque

#### Logging
from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

class RNNAgent(object):
    def __init__(self, action_space):
        
        ### model hyperparameters

        self.epsilon = 0.9  # how much do we explore initially
        self.epsilon_decay_rate = 0.95  # rate by which exploration decreases, used for constant epsilon decay strategy
        self.high_score = 0  # keep track of highest score obtained thus far
        self.did_well_threshold = 0.80  # how close we need to be to our high score to have "done well"
        self.network_has_had_training = False  # has our neural net had any training
        self.last_good_batch = tuple()  # memory for the last good episode we eperienced 
        
        self.experience = 0  # integer for keeping track of how much good experience we've had, used in custom epsilon decay function
            
        self.last_100_episode_scores = deque(maxlen = 100) # keep track of average score from last 100 episodes

    """
       Function for letting us know if we did well based on the rewards received this episode and the 
       did_well_threshold parameter.
    """
    def did_we_do_well(self, episode_rewards):

        if episode_rewards > self.did_well_threshold * self.high_score:

            return True

        return False



    """
       Function for adding an experience memory from episodes were we've "done well".
    """
    def add_to_experience(self, episode_length):

        self.experience += episode_length



    """
        Function that updates our highest score acheived thus far.
    """
    def update_high_score(self, episode_rewards):

        if episode_rewards > self.high_score:

            self.high_score = episode_rewards

 

    """
        Typical epsilon decay function.
    """
    def decay_epsilon(self):

        self.epsilon *= self.epsilon_decay_rate



    """
        Customized epsilon decay function.
    """
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

    

    """
        Function for deriving epsilon from the uncertainty reflected in our LSTMs action output.
    """
    def epsilon_from_uncertainty(self, unscaled_confidence):

            self.epsilon = 0.1  # Initialize epsilon

            ## Update epsilon if we aren't very confident
            if unscaled_confidence < 1.5: 
                confidence = unscaled_confidence / 1.5
                uncertainty = min(0.9, 1.0 - confidence)  # Uncertainty is the opposite of confidence, limit to max of 0.9
                self.epsilon = max(uncertainty, 0.1)  # Limit to min of 0.1  

sess = tf.InteractiveSession()

nb_inputs = 4
nb_neurons = 30
nb_outputs = 2
nb_timesteps = 1
nb_layers = 3

## Data
X_batch = np.array([
    [[0, 1, 2, 5], [9, 8, 7, 4]], # Batch 1
    [[3, 4, 5, 2], [0, 0, 0, 0]], # Batch 2
    [[6, 7, 8, 5], [6, 5, 4, 2]], # Batch 3
])

# define the shape of the data placeholder (tensor)
state = tf.placeholder(tf.float32, [None, nb_timesteps, nb_inputs])
actions = tf.placeholder(tf.int32, [None])

# define network
basic_lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=nb_neurons)
learning_rate = 0.001

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=nb_neurons) for layer in range(nb_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

outputs, states = tf.nn.dynamic_rnn(multi_cell, state, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, nb_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, actions, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess.run(tf.initialize_all_variables())

## Logging
loss_summary = tf.summary.scalar('Loss', loss)
#file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

env = gym.make("CartPole-v0")
wondering_gnome = RNNAgent(env.action_space)

for i_episode in range(10000):

    observation = env.reset()
    episode_rewards = 0
    episode_states_list = []
    episode_actions_list = []

    for t in range(200):
        env.render()

        state_for_mem = observation
        current_state = np.expand_dims(observation, axis=0)
        current_state = current_state.reshape(1,nb_timesteps,nb_inputs)
        action = env.action_space.sample()

        if wondering_gnome.network_has_had_training:
            random_fate = np.random.random()
            raw_output = logits.eval(feed_dict={state: current_state})

            if random_fate > wondering_gnome.epsilon:
                action = np.argmax(raw_output)

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
    train_step.run(feed_dict={state: wondering_gnome.last_good_batch[0], actions: wondering_gnome.last_good_batch[1]}) # , keep_prob: 0.75})

    ## Our LSTM has been trained
    if i_episode > 1:
        wondering_gnome.network_has_had_training = True

    ## Do some logging of our current loss
    if i_episode % 20 == 0:
        summary_str = loss_summary.eval(feed_dict={state: wondering_gnome.last_good_batch[0], actions: wondering_gnome.last_good_batch[1]})
        step = i_episode
        #file_writer.add_summary(summary_str, step)