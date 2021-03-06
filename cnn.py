
##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   cnn.py - Implementation of CNN networks                ##
##                                                          ##
##############################################################
##############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


############################ CONVOLUTIONAL NEURAL NETWORK DEFINITION ###############################################################
#Class with 7 different CNN architectures
class ConvNetActorCritic(nn.Module):
    def __init__(self, img_num, num_inputs, num_outputs, hidden_ac, hidden_cnn, out_cnn, cnn_type, std=0.0):
        super().__init__()

        self.type = cnn_type

        if self.type == 1:
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.conv11 = nn.Conv2d(6, 6, 5)
            self.pool4 = nn.AvgPool2d(4, 4)
            self.conv2= nn.Conv2d(6, 16, 5)

            if img_num == 2:
                self.fc2 = nn.Linear(1344, hidden_cnn)
            else:
                self.fc2 = nn.Linear(576, hidden_cnn)
            self.fc3 = nn.Linear(hidden_cnn, out_cnn)

        elif self.type == 2:
            self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.conv11 = nn.Conv2d(6, 12, 5, 1, 2)
            self.pool4 = nn.AvgPool2d(2, 2)
            self.conv2= nn.Conv2d(12, 24, 5, 1, 2)

            if img_num == 2:
                self.fc2 = nn.Linear(3072, hidden_cnn)
            else:
                self.fc2 = nn.Linear(1536, hidden_cnn)
            self.fc3 = nn.Linear(hidden_cnn, out_cnn)

        elif self.type == 3:
            self.conv1 = nn.Conv2d(3, 24, 5, 1, 2)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.conv11 = nn.Conv2d(24, 30, 5, 1, 2)
            self.pool4 = nn.AvgPool2d(4, 4)
            if img_num == 2:
                self.fc2 = nn.Linear(3840, hidden_cnn)
            else:
                self.fc2 = nn.Linear(1920, hidden_cnn)
            self.fc3 = nn.Linear(hidden_cnn, out_cnn)

        elif self.type == 4:
            self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.conv11 = nn.Conv2d(6, 12, 5, 1, 2)
            self.pool4 = nn.AvgPool2d(2, 2)
            self.conv2= nn.Conv2d(12, 18, 5, 1, 2)
            self.conv22= nn.Conv2d(18, 32, 5, 1, 2)
            if img_num == 2:
                self.fc2 = nn.Linear(1024, hidden_cnn)
            else:
                self.fc2 = nn.Linear(512, hidden_cnn)
            self.fc3 = nn.Linear(hidden_cnn, out_cnn)

        elif self.type == 5:
            self.conv1 = nn.Conv2d(3, 8, 5, 1, 2)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.conv11 = nn.Conv2d(8, 8, 5, 1, 2)
            self.pool4 = nn.AvgPool2d(4, 4)

            if img_num == 2:
                self.fc2 = nn.Linear(1024, hidden_cnn)
            else:
                self.fc2 = nn.Linear(512, hidden_cnn)
            self.fc3 = nn.Linear(hidden_cnn, out_cnn)


        elif self.type == 6:
            self.conv1 = nn.Conv2d(3, 12, 5, 1, 2)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.conv11 = nn.Conv2d(12, 24, 5, 1, 2)
            self.pool4 = nn.AvgPool2d(2, 2)
            self.conv2= nn.Conv2d(24, 48, 5, 1, 2)
            self.conv22= nn.Conv2d(48, 48, 5, 1, 2)

            if img_num == 2:
                self.fc2 = nn.Linear(1536, hidden_cnn)
            else:
                self.fc2 = nn.Linear(768, hidden_cnn)
            self.fc3 = nn.Linear(hidden_cnn, out_cnn)


        elif self.type == 7:
            self.conv1 = nn.Conv2d(3, 8, 5, 1, 2)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.conv11 = nn.Conv2d(8, 8, 5, 1, 2)
            self.pool4 = nn.AvgPool2d(4, 4)
            self.conv2= nn.Conv2d(8, 8, 5, 1, 2)

            if img_num == 2:
                self.fc2 = nn.Linear(1024, hidden_cnn)
            else:
                self.fc2 = nn.Linear(512, hidden_cnn)
            self.fc3 = nn.Linear(hidden_cnn, out_cnn)

#################################################################
 
        self.critic = nn.Sequential(
            nn.Linear(num_inputs + out_cnn, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs + out_cnn, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, num_outputs),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)


    def forward(self, sensor, x, detach_cnn=False):

        if self.type == 1:
            x = F.relu(self.conv1(x))
            x = self.pool2(F.relu(self.conv11(x)))
            x = self.pool4(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.type == 2:
            x = self.pool2(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv11(x)))
            x = self.pool4(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.type == 3:
            x = self.pool2(F.relu(self.conv1(x)))
            x = self.pool4(F.relu(self.conv11(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)


        elif self.type == 4:
            x = self.pool2(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv11(x)))
            x = self.pool4(F.relu(self.conv2(x)))
            x = self.pool4(F.relu(self.conv22(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.type == 5:
            x = self.pool2(F.relu(self.conv1(x)))
            x = self.pool4(F.relu(self.conv11(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.type == 6:
            x = self.pool2(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv11(x)))
            x = self.pool4(F.relu(self.conv2(x)))
            x = self.pool4(F.relu(self.conv22(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.type == 7:
            x = self.pool2(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv11(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

#####################################################################

        x_aug = torch.hstack( (x, sensor) )
        value = self.critic(x_aug)

        if detach_cnn:
            x = x.detach()
            x_aug = torch.hstack( (x, sensor) )

        mu    = self.actor(x_aug)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)

        return dist, value