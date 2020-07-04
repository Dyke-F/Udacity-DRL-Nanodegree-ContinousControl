# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import copy
from collections import namedtuple, deque

from TD3_Actor_Critic_Model import Actor, Critic
from replaybuffer import ReplayBuffer
from ornstein_uhlenbeck_noise import OUNoise

# get hyperparameters
from Hyperparameters import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# implement TD3 
class TD3():
    """ Twin Delayed Deep Deterministic Policy Gradient Model """

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
                self.noise_decay = NOISE_DECAY
                
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
                self.soft_update(self.actor_local, self.actor_target, 1)
                self.soft_update(self.critic_local, self.critic_target, 1)
                
                self.learn_counter = 0
                
                
    def act(self, state, add_noise=True):
        """ Choose an action while interacting and learning in the environment """

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * self.noise_decay
            self.noise_decay *= self.noise_decay
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, noise_clip=0.5, policy_freq=2):
        """ Sample from experiences and learn """

        # update the learn counter
        self.learn_counter += 1

        # get experience tuples
        states, actions, rewards, next_states, dones  = experiences
            
        # build noise on the action 
        ##### CAVE: need to put actions onto cpu() to create a cpu tensor that is put onto CUDA with .to(device)
        #noise = torch.FloatTensor(actions.cpu()).data.normal_(0, policy_noise).to(device)
        #noise = noise.clamp(-noise_clip, noise_clip)
        ### <<--- adding this kind of noise was implemented in the paper on github,
        ### but i used OU-Noise in the act method, so maybe better to use the same while learning

        noise = torch.FloatTensor([self.noise.sample() for _ in range(len(actions))]).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)  
        # clip between -/+ max action dims because action+noise might run oor
        next_action = (self.actor_target(next_states) + noise).clamp(-1, 1)

        # compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_states, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (gamma * target_Q * (1-dones)).detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic_local(states, actions)

        # compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # update the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # delay the policy update
        if self.learn_counter % policy_freq == 0:
                    
                # get actor_local predicted next action and use critic_local to complete
                actions_pred = self.actor_local.forward(states)
                actor_loss = -self.critic_local.Q1(states, actions_pred).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # delay update of actor and critic target models
                self.soft_update(self.actor_local, self.actor_target, TAU)
                self.soft_update(self.critic_local, self.critic_target, TAU)


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
           
   
