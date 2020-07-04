import torch
import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer:
    """ Fixed-size buffer for storing experience tuples for sample-learning in
        the TD3 agent 
    """
    
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """ Initialize Replay Buffer object.
        ==========
        PARAMETERS s
        ==========
            action_size (int) = dimension of each action
            buffer_size (int) = maxlen of the buffer storage
            batch_size (int) = number of samples collected randomly at each learn step
            seed (int) = random seed
        """
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        
        self.device = device

    def add(self, states, actions, rewards, next_states, dones):
        """ Append a new experience to memory """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)


    def sample(self):
        """ Randomly collect batch_size of experiences for learning step """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """ Returns current size of internal ReplayBuffer memory """
        return len(self.memory)

