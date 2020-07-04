# imports
import copy
import random
import sys
import numpy as np

# create noise
class OUNoise(object):
    """ Ornstein Uhlenbeck noise process """
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """ Initialize parameters and noise process """
        
        # size = action_size; noise will be added to network outputs (action)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        """ Reset the internal state (=noise) to the mean (mu) """
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """ Update the internal state and return it as a noise sample to the action
            PSEUDOCODE RETURNS
            ==================
                New state = Old State + Theta * (Mu - Old State) + Sigma * (# Random values as long as state)
        """
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
