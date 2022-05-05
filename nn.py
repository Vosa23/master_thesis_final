##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   nn.py - Implementation of Actor-Critic networks        ##
##                                                          ##
##############################################################
##############################################################


import torch
import torch.nn as nn
from torch.distributions import Normal


############################ NEURAL NETWORK DEFINITION #######################################################################
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, num_layers, std=0.0):
        super().__init__()

        if num_layers == 2:
            self.critic = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            
            self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
                nn.Tanh()
            )
        elif num_layers == 1:
            self.critic = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

            self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
                nn.Tanh()
            )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)


    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

    #Multilayer perceptron
    def mlp(self, sizes, activation, output_activation=nn.ReLU()): #nn.Identity
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)