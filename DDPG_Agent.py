# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import copy
from collections import namedtuple, deque

from DDPG_Actor_Critic_Model import Actor, Critic
from replaybuffer import ReplayBuffer
from ornstein_uhlenbeck_noise import OUNoise

# get hyperparameters
from Hyperparameters import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# implement DDPG 
class DDPG():
    """ Deep Deterministic Policy Gradient Model """

    def __init__(self, state_size, action_size, random_seed):
        """ Initialize the model with arguments as follows:
                
                    ARGUMENTS
                    =========
                        - state_size (int) = dimension of input space
                        - action_size (int) = dimension of action space
                        - random_seed (int) = random seed

                    Returns 
                    =======
                        - best learned action to take after Actor-Critic Learning
         """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # create noise
        self.noise = OUNoise(action_size, random_seed)
                
        # create memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, device)
                


        # Actor Networks (local online net + target net)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)

        # Critic Networks (local online net + target net)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
                
        # instantiate online and target networks with same weights
        self.hard_update(self.actor_local, self.actor_target,)
        self.hard_update(self.critic_local, self.critic_target)
    
    
    def hard_update(self, local, target):
        for local_param, target_param in zip(local.parameters(), target.parameters()):
            target_param.data.copy_(local_param.data)
                
                
    def act(self, state, add_noise=True):
        """ Choose an action while interacting and learning in the environment """

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        # Perform soft update of the target networks
        # at every time step, keep 1-tau of target network
        # and add only a small fraction (tau) of the current online networks
        # to prevent oszillation
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        # at every iteration, add new SARS' trajectory to memory, then learn from batches 
        # if learning_step is reached and enough samples are in the buffer
        
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
           
   
