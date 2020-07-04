# import PyTorch and NumPy modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# get hyperparameters
from Hyperparameters import *


# reset hidden parameters
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


# create an Actor class
class Actor(nn.Module):
    """ Create an Actor Policy model, that maps state to action """
    
    def __init__(self, state_size, action_size, seed,
                 fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        
                """ ARGUMENTS
                    =========
                        - state_size (int) = dimension of input space
                        - action_size (int) = dimension of action space
                        - seed (int) = random seed
                        - fc[1,2,...,X]_units (int) = number of neurons per layer
                        
                    RETURNS
                    =======
                        - Mapping of input space to action probability

                """

                super(Actor, self).__init__()
                self.seed = torch.manual_seed(seed)
                self.fc1 = nn.Linear(state_size, fc1_units)
                self.fc2 = nn.Linear(fc1_units, fc2_units)
                self.fc3 = nn.Linear(fc2_units, action_size)
                
                self.bn1 = nn.BatchNorm1d(fc1_units)
                self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        
        # Map a state -> action probability using the policy
        
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
            
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# Create a Critic class
class Critic(nn.Module):
    """ Critic Value-Q-Network Model """

    def __init__(self, state_size, action_size, seed, fcs1_units=FCS1_UNITS, fc2_units=FC2_UNITS):
        
        """ ARGUMENTS
            =========
                - state_size (int) = dimension of input space
                - action_size (int) = dimension of action space
                - seed (int) = random seed
                - fc[s1,2,...,X]_units (int) = number of neurons per layer
                
            Returns 
            =======
                - Mapping of state + action -> Q-values
        """

    # initialize
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
    # define network architecture
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    # add batch norm
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        
    # reset self parameters
        self.reset_parameters()

        
    def reset_parameters(self):
        
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, state, action):
        """ Critic Network maps states and actions to Q-values """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
 
        xs1 = F.relu(self.fcs1(state))
        x1 = self.bn1(xs1)
        x1 = torch.cat((x1, action), dim=1)
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        return x1
