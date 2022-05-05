
##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   run.py                                                 ##
##   Based on                                               ##
##############################################################
##############################################################


import sys
import logging
from yaml import Loader,Dumper, load, dump
# from simulation import Simulation

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

import optimization #used to be ppo_cnn
import ppo_cnn
# import ppo_old
# import ppo2
# import ppo3

import logging

log = logging.getLogger(__name__)

class Simulation:

    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_vars = [ k for (k,v) in self.cfg['sensors'].items() if v ]
        self.obs_preprocess = {k:v for (k,v) in self.cfg['sensors'].items() if v }


    def run(self):

        choice = self.cfg['simulation']['algorithm']


        if choice == 'ppo':
            if self.cfg['simulation']['architecture'] == 'cnn':
                optimization.run(self.cfg, self.obs_vars, self.obs_preprocess)
            else:
                optimization.run(self.cfg, self.obs_vars, self.obs_preprocess)
        elif choice == 'ppo_old':
            ppo_old.run(self.cfg)
        elif choice == 'ppo_cnn':
            ppo_cnn.run(self.cfg, self.obs_vars, self.obs_preprocess)
        elif choice == 'ppo2_basic':
            ppo2.run_basic(self.cfg)
        elif choice == 'ppo3':
            ppo3.run(self.cfg, self.obs_vars, self.obs_preprocess)
        elif choice == 'ppo3_basic':
            ppo3.run_basic(self.cfg)
        else:
            log.info("Nothing chosen..see config (simulation:algorithm)")


class CnnToSensors(nn.Module):
    def __init__(self):
        super(CnnToSensors, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.pool2 = nn.AvgPool2d(2, 2) 
        self.conv11 = nn.Conv2d(6, 12, 5, 1, 2)
        self.pool4 = nn.AvgPool2d(2, 2)
        self.conv2= nn.Conv2d(12, 24, 5, 1, 2)
        # self.pool44 = nn.AvgPool2d(2, 2)
        # if img_num == 2:
        #     self.fc2 = nn.Linear(1344, hidden_cnn)      #1344 #TODO DVO maybe this will have to be changed -> if img_num == 2 WILL BE MORE..
        # else:
        self.fc2 = nn.Linear(1536, 512)          #used to be 90...instead of 256    COMPARE PAK s temi 90 (run 21-12-12 cas)
        self.fc3 = nn.Linear(512, 19)               #+ jeste bez LIDARu to zkusit    #120 misto 60 pro 1 img


    def forward(self, x):
        # print(x.shape)
        x = self.pool2(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv11(x)))
        # print(x.shape)
        x = self.pool4(F.relu(self.conv2(x)))

        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)

        return x

#CNN TO SENSORS OPTION...so change based on the Google Colab NETWORK.
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv11 = nn.Conv2d(6, 6, 5)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.conv2= nn.Conv2d(6, 16, 5)

        self.fc2 = nn.Linear(576, 512)
        self.fc3 = nn.Linear(512, 20)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool2(F.relu(self.conv11(x)))
        x = self.pool4(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

#################################################################################################################################

logging.basicConfig(filename='logger.trc',
                    filemode='a',
                    format='%(asctime)s:%(msecs)-4d %(levelname)-6s %(message)-60s %(name)-13s %(funcName)-20s %(pathname)s',
                    datefmt='%d/%m/%H:%M:%S',
                    level=logging.DEBUG)


log = logging.getLogger(__name__)

def parser( args ):

    if len(args) >= 2:
        if args[1] == '-h':
            print('Help, run as: python3 run.py -c config.yaml')
            sys.exit(0)
        elif args[1] == '-c' and len(args) == 3:
            cfg = args[2]
        else:
            print("Invalid option. Run -h for help")
            sys.exit(-1)
    else:
        print("No option. Run -h for help")
        sys.exit(0)


    try:
        #Loading the config
        with open(cfg, 'r') as cfg:
            config = load(cfg, Loader=Loader )
            dump(config, Dumper=Dumper )

        #Loading the track info
        with open(config['simulation']['tracks_config'], 'r') as cfg_track:
            tracks_config = load(cfg_track, Loader=Loader )
            track = config['setup']['track']
            config['setup']['track_property'] = tracks_config[track]
    except Exception as e:
        print( e )
        sys.exit(-1)

    else:
        return config


def main():

    config = parser(sys.argv)
    sim = Simulation(config)
    sim.run()

if __name__ == '__main__':
    main()
