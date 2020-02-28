import gym
import random
import copy
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

class PyTorchAgent(nn.Module):
    """description of class"""
    def __init__(self):
        #Create model
        self.model = "Not implemented"

    def remember(self, oldState, action, reward, state):
        raise NotImplementedError

    def get_action(self, screen):
        raise NotImplementedError

    def take_action(self):
        raise NotImplementedError

if __name__ == "__main__":
    main()