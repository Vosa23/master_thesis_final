
##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   torcs_env.py - client wrapper for TORCS                ##
##   Based on: https://github.com/dosssman/GymTorcs         ##
##############################################################
##############################################################

import gym
from gym import spaces
import numpy as np

import gym_torcs.snakeoil3_gym as snakeoil3
import numpy as np
import copy
import os
import subprocess
import psutil
import random
import scipy.stats

import logging
log = logging.getLogger(__name__)

DEF_BOX_DTYPE = np.float32

class TorcsEnv( gym.Env):

    log.info('init TorcsEnv object')

    terminal_judge_start = 50           # After this step the penalty for small progress is applied
    termination_limit_progress = 1      # [km/h], episode terminates if car is running slower than this limit
    default_speed = 300.

    initial_reset = True

    def __init__(self, vision=False,
        throttle=False,
        gear_change=False,
        race_config_path=None,
        race_speed=1.0,
        rendering=True,
        no_damage=False,
        lap_limiter=1,
        recdata=False,
        noisy=False,
        rec_episode_limit=1,
        rec_timestep_limit=3600,
        rec_index=0,
        hard_reset_interval=11,
        randomisation=False,
        profile_reuse_ep=500,
        rank=0,
        obs_vars=None,
        obs_preprocess_fn=None,
        obs_normalization=False,
        reward_function_used=1,
        track_dict=None,
        config=None
        ):

        # Define mins and maxes for each obseration
        self.obs_maxima = { 
            'angle':                np.pi,
            'curLapTime':           np.Inf,
            'damage':               10000,
            'distFromStart':        np.Inf,
            'totalDistFromStart':   np.Inf,
            'distRaced':            np.Inf,
            'focus':                200.,
            'fuel':                 np.Inf,
            'lap':                  100,
            'gear':                 6,
            'lastLapTime':          np.Inf,
            'opponents':            200.,
            'racePos':              36,
            'rpm':                  10000,
            'speedX':               300.,
            'speedY':               300.,
            'speedZ':               300.,
            'track':                200.,
            'trackPos':             1,
            'wheelSpinVel':         100,
            "z":                    np.Inf,
            #"img":                 255,
        }

        self.obs_minima = {
            'angle':                -np.pi,
            'curLapTime':           0.,
            'damage':               0.,
            'distFromStart':        0.,
            'totalDistFromStart':   0.,
            'distRaced':            0.,
            'focus':                0.,
            'fuel':                 0.,
            'lap':                  0.,
            'gear':                 -1,
            'lastLapTime':          0.,
            'opponents':            0.,
            'racePos':              1.,
            'rpm':                  0.,
            'speedX':               0.,
            'speedY':               0.,
            'speedZ':               0.,
            'track':                0.,
            'trackPos':             -1,
            'wheelSpinVel':         0.,
            "z":                    np.NINF,
            #"img":                 0,

        }
        self.obs_dtypes = {
            'angle':                np.float32,
            'curLapTime':           np.float32,
            'damage':               np.float32,
            'distFromStart':        np.float32,
            'totalDistFromStart':   np.float32,
            'distRaced':            np.float32,
            'focus':                np.float32,
            'fuel':                 np.float32,
            'lap':                  np.uint8,
            'gear':                 np.uint8,
            'lastLapTime':          np.float32,
            'opponents':            np.float32,
            'racePos':              np.uint8,
            'rpm':                  np.float32,
            'speedX':               np.float32,
            'speedY':               np.float32,
            'speedZ':               np.float32,
            'track':                np.float32,
            'trackPos':             np.float32,
            'wheelSpinVel':         np.float32,
            "z":                    np.float32,
            #"img":                 np.uint8
        }
        self.obs_dim = {
            'angle':                1,
            'curLapTime':           1,
            'damage':               1,
            'distFromStart':        1,
            'totalDistFromStart':   1,
            'distRaced':            1,
            'focus':                5,
            'fuel':                 1,
            'lap':                  1,
            'gear':                 1,
            'lastLapTime':          1,
            'opponents':            36,
            'racePos':              1,
            'rpm':                  1,
            'speedX':               1,
            'speedY':               1,
            'speedZ':               1,
            'track':                19,
            'trackPos':             1,
            'wheelSpinVel':         4,
            "z":                    1,
            #"img":                 4096
        }

        self.obs_vars = obs_vars

        # Set default observation preprocessing method
        self.obs_normalization = obs_normalization
        self.obs_preprocess_fn = obs_preprocess_fn

        #DVO custom track selection
        self.track_dict = track_dict

        #DVO load the cfg file
        self.config = config

        # Set the default raceconfig file
        if race_config_path is None:
            if self.track_dict is None:
                #Track::Michigan Oval
                race_config_path = os.path.join( os.path.dirname(os.path.realpath(__file__)),
                "raceconfigs/default.xml")
            else:
                race_config_path = os.path.join( os.path.dirname(os.path.realpath(__file__)),
                "raceconfigs/" + str(self.track_dict['name']) + ".xml")

        self.seed_value = 42

        high = np.hstack([ [self.obs_maxima[obs_name]] * self.obs_dim[obs_name] for obs_name in self.obs_vars])
        low = np.hstack([ [self.obs_minima[obs_name]] * self.obs_dim[obs_name] for obs_name in self.obs_vars])

        self.observation_space = spaces.Box( low=low, high=high, dtype=DEF_BOX_DTYPE)

        # Action spaces
        if throttle and gear_change:
            self.action_space = spaces.Box( low=np.array( [-1., -1., -1]),
                high=np.array( [1., 1., 6]),
                dtype=[np.float32, np.float32, np.int32])
        elif throttle:
            # Steering and accel / decel
            self.action_space = spaces.Box( low=np.array( [-1, 0.0]),
                high=np.array( [1, 1.]), dtype=np.float32)
        else:
            # Steering only
            self.action_space = spaces.Box( low=np.array( [-1.]),
                high=np.array( [1.]), dtype=np.float32)


        # Support for blackbox optimal reset
        self.reset_ep_count = 1
        self.hard_reset_interval = hard_reset_interval


        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.race_speed = race_speed
        self.rendering = rendering
        self.no_damage = no_damage
        self.recdata = recdata
        self.noisy = noisy
        self.reward_function_used=reward_function_used

        # Track randomization related
        self.randomisation = randomisation
        self.profile_reuse_count = 0
        self.profile_reuse_ep = profile_reuse_ep

        # Default
        self.initial_run = True

        # Raceconfig compat edit
        self.torcs_process_id = None
        self.race_config_path = race_config_path

        # Paralelization support
        self.rank = rank
        self.server_port = 3000 + self.rank
        # For one server instance, only one client supported

        # Freshly initialised
        if self.randomisation:
            self.randomise_track()

        # Internal time tracker for
        # The episode will end when the lap_limiter is reached
        # To put it simply if you want env to stap after 3 laps, set this to 4
        # Make sure to run torcs itself for more than 3 laps too, otherwise,
        # before terminating the episode
        self.lap_limiter = lap_limiter
        self.rec_episode_limit = rec_episode_limit
        self.rec_timestep_limit = rec_timestep_limit
        self.rec_index = rec_index

        args = ["torcs", "-nofuel", "-nolaptime",
            "-a", str( self.race_speed)]


        if self.no_damage:
            args.append( "-nodamage")

        if self.noisy:
            args.append( "-noisy")

        if self.vision:
            args.append( "-vision")

        if not self.rendering:
            args.append( "-T") # Run in console

        if self.race_config_path is not None:
            args.append( "-raceconfig")
            args.append( self.race_config_path)

        if self.track_dict is not None:
            args.append( '-track ' + str(self.track_dict['name']) )

        if self.recdata:
            args.append( "-rechum %d" % self.rec_index)
            args.append( "-recepisodelim %d" % self.rec_episode_limit)
            args.append( "-rectimesteplim %d" % self.rec_timestep_limit)

        # For parallelization support
        args.append( "-p %d" % self.server_port)
        args.append("&")

        self.torcs_process_id = subprocess.Popen( args, shell=False).pid


    def seed( self, seed_value=42):
        self.seed_value = seed_value


    def step(self, u):

        #Packet loss
        if self.check_on_fail():
            u = np.zeros((u.shape))

        log.info('making action, step')

        client = self.client

        #Convert thisAction to the actual torcs actionstr
        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            if this_action['accel'] >= 0.0:
                action_torcs['accel'] = this_action['accel']
                action_torcs['brake'] = 0.0
            else:
                action_torcs['brake'] = -this_action['accel']
                action_torcs['accel'] = 0.0

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d
        
        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs, obs_pre)


        # Reward setting

        track       = np.array(obs['track'])
        trackPos    = np.array(obs['trackPos'])
        speedX      = np.array(obs['speedX'])
        opponents   = np.array(obs['opponents'])
        angle       = np.array(obs['angle'])
        damage      = np.array(obs['damage'])
        damage_pred = np.array(obs_pre['damage'])

        road_width = np.float(self.track_dict['width'])


        #### REWARD FUNCTION R1 #######################
        if self.reward_function_used == 1:
            progress = speedX*np.cos(angle) - np.abs(speedX*np.sin(angle)) - speedX * np.abs(trackPos)


        #### REWARD FUNCTION R2 #######################
        elif self.reward_function_used == 2:
            progress = speedX*( np.cos(angle) - np.abs(trackPos) )


        #### REWARD FUNCTION R3 #######################
        elif self.reward_function_used == 3:
            progress = speedX * (np.cos(angle) - (1 / ( 1 + np.exp(-4* np.abs(trackPos) - 0.5 * road_width ) ) ) )


        #### REWARD FUNCTION R4 #######################         # FOR RACING MODE
        elif self.reward_function_used == 4:
            oppo_dist = np.min(opponents)
            reward_car_distance = -10. if oppo_dist <= 5. else 0.
            progress = speedX* np.cos(angle) - np.abs(speedX*np.sin(angle)) - speedX * np.abs(trackPos)  + reward_car_distance

            # OVERTAKING
            reward = 0
            if obs["racePos"] < obs_pre["racePos"]:
                reward += 100.
            if obs["racePos"] > obs_pre["racePos"]:
                reward -= 100.


        #Not doing anything penalty
        reward = progress - 1.

        # COLLISION
        if damage - damage_pred > 0.:
            reward += -100.
            # terminate_episode = True

        terminate_episode = False

        # OUT OF TRACK
        if (abs(track.any()) > 1 or abs(trackPos) > 1):
            reward += -70.
            terminate_episode = True

        # PROGRESS TOO SMALL
        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                reward += -10.
                if self.config['simulation']['mode'] == 'eval':
                    terminate_episode = True            

        # GOES BACKWARDS
        if np.cos(angle) < 0.:
            reward += -80.
            terminate_episode = True

        client.R.d['meta'] = False
        self.time_step += 1

                      
        mini_batch = 32
        if self.config is not None:
            mini_batch = self.config['ppo']['MINI_BATCH_SIZE']

         #Send a reset signal
        if terminate_episode is True and self.time_step > mini_batch:
            self.initial_run = False
            client.R.d['meta'] = True
            client.respond_to_server()

        return self.get_obs(), np.array(reward), client.R.d['meta'], {}




    def reset(self, relaunch=False):

        self.time_step = 0
        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            #Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True or self.reset_ep_count % self.hard_reset_interval == 0:
                self.reset_torcs()
                self.reset_ep_count = 1
                print("### TORCS is RELAUNCHED ###")

        if self.randomisation:
            self.randomise_track()

        print("Snakeoil3.Client")
        self.client = snakeoil3.Client(p=self.server_port, vision=self.vision,
            process_id=self.torcs_process_id,
            race_config_path=self.race_config_path,
            race_speed=self.race_speed,
            rendering=self.rendering, lap_limiter=self.lap_limiter,
            damage=self.no_damage, recdata=self.recdata, noisy=self.noisy,
            rec_index = self.rec_index,rec_episode_limit=self.rec_episode_limit,
            rec_timestep_limit=self.rec_timestep_limit, rank=self.rank,
            trackname = self.track_dict['name'] )  #Open new UDP in vtorcs

        self.client.MAX_STEPS = np.inf

        client = self.client

        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs, None)

        self.last_u = None
        self.initial_reset = False

        self.torcs_process_id = self.client.torcs_process_id

        self.reset_ep_count += 1
        self.profile_reuse_count += 1

        return self.get_obs()


    def end(self):
        if self.torcs_process_id is not None:
            try:
                p = psutil.Process( self.torcs_process_id)
                for pchild in p.children():
                    pchild.terminate()

                p.terminate()
            except Exception:
                self.torcs_process_id = None


    def close(self):
        self.end()

    def get_obs(self):
        return self.observation


    def reset_torcs(self):
        print( "Process PID: ", self.torcs_process_id)
        if self.torcs_process_id is not None:
            try:
                p = psutil.Process( self.torcs_process_id)
                for pchild in p.children():
                    pchild.terminate()

                p.terminate()
            except Exception:
                pass

        if self.randomisation:
            self.randomise_track()

        print("LOGGING: torcs -nofuel ( reset_torcs() ).")
        args = ["torcs", "-nofuel", "-nolaptime",
            "-a", str( self.race_speed)]

        if self.no_damage:
            args.append( "-nodamage")

        if self.noisy:
            args.append( "-noisy")

        if self.vision:
            args.append( "-vision")

        if not self.rendering:
            args.append( "-T") # Run in console

        if self.race_config_path is not None:
            args.append( "-raceconfig")
            args.append( self.race_config_path)

        if self.recdata:
            args.append( "-rechum %d" % self.rec_index)
            args.append( "-recepisodelim %d" % self.rec_episode_limit)
            args.append( "-rectimesteplim %d" % self.rec_timestep_limit)

        args.append( "-p %d" % self.server_port)
        args.append("&")

        self.torcs_process_id = subprocess.Popen( args, shell=False).pid


    #Setting the action value for the server
    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[1]})

        if self.gear_change is True:
            torcs_action.update({'gear': u[2]})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec, pre_obs_image_vec=None):
        image_vec = np.array(obs_image_vec)

        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = r.reshape(sz)
        g = g.reshape(sz)
        b = b.reshape(sz)
        
        img = np.array([r, g, b], dtype=np.uint8)
        img = img[np.newaxis,:,:,:]
        
        if self.config['simulation']['img_num'] == 2 or self.config['simulation']['img_type'] == 'diff':
            pre_image_vec = np.array(pre_obs_image_vec)

            pre_r = pre_image_vec[0:len(pre_image_vec):3]
            pre_g = pre_image_vec[1:len(pre_image_vec):3]
            pre_b = pre_image_vec[2:len(pre_image_vec):3]

            sz = (64, 64)
            pre_r = pre_r.reshape(sz)
            pre_g = pre_g.reshape(sz)
            pre_b = pre_b.reshape(sz)
            
            pre_img = np.array([pre_r, pre_g, pre_b], dtype=np.uint8)
            pre_img = pre_img[np.newaxis,:,:,:]


            if self.config['simulation']['img_num'] == 2 and self.config['simulation']['img_type'] == 'reg':
                two_img = np.concatenate((img, pre_img), axis=3)
                return two_img
            elif self.config['simulation']['img_num'] == 2 and self.config['simulation']['img_type'] == 'diff':
                diff_img = np.abs(img - pre_img)
                two_diff_img = np.concatenate((img, diff_img), axis=3)
                return two_diff_img
            elif self.config['simulation']['img_num'] == 1 and self.config['simulation']['img_type'] == 'diff':
                diff_img = np.abs(img - pre_img)
                return diff_img
        else:
            return img
        
        
    def make_observaton(self, raw_obs, pre_raw_obs=None):
        dict_obs = {}

        for obs_name in self.obs_vars:

            if self.obs_normalization:
                dict_obs[obs_name] = np.array(raw_obs[obs_name], dtype=self.obs_dtypes[obs_name])/(self.obs_maxima[obs_name] - self.obs_minima[obs_name])

                #do not normalize this metric
                dict_obs['distRaced'] = raw_obs['distRaced']
                dict_obs['damage'] = raw_obs['damage']

            else:
                dict_obs[obs_name] = raw_obs[obs_name]

        #USING CNN for img
        if self.config['simulation']['architecture'] == 'cnn' or self.config['simulation']['cnn_to_sensors'] or self.config['simulation']['record_samples']:
            
            if self.config['simulation']['img_num'] == 2 or self.config['simulation']['img_type'] == 'diff':
                if pre_raw_obs is not None:
                    image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'], pre_raw_obs['img'])
                else:
                    image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'], raw_obs['img'])    #First iter of (img diff) version
            else:
                image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])
            
            if self.obs_normalization:
                image_rgb = image_rgb/255

                #standardization - can be on single images, not batch? bcs of the mean and std...
                #image_rgb = (image_rgb - image_rgb.mean() ) / image_rgb.std()
            
            #Packet loss
            if self.check_on_fail():
                for _,v in dict_obs.items():
                    v = 0. if isinstance(v, float) else np.zeros((v.shape))
                image_rgb = np.zeros((image_rgb.shape))

            #Noise/Fail of sensor
            elif self.check_on_noise():
                for _,v in dict_obs.items():
                    if random.random() < 0.5:
                        v += np.random.normal(0,0.1, None if isinstance(v, float) else v.size)
                if random.random() < 0.5:
                    img_noise = np.random.normal(0,0.1, image_rgb.size )
                    img_noise = np.reshape(img_noise, newshape=image_rgb.shape)
                    image_rgb += img_noise
            
            # print(image_rgb)
            return (dict_obs, image_rgb)

        #Packet loss
        if self.check_on_fail():
            for _,v in dict_obs.items():
                v = 0. if isinstance(v, float) else np.zeros((v.shape))

        #Noise/Fail of sensor
        elif self.check_on_noise():
            for _,v in dict_obs.items():
                if random.random() < 0.5:
                    v += np.random.normal(0,0.1, None if isinstance(v, float) else v.size )

        return dict_obs


    def check_on_fail(self):
        if self.config['simulation']['comm_failure']:
            rand_val = random.gauss(0,1)
            perc_l = scipy.stats.norm.ppf(self.config['simulation']['comm_prob'], loc=0, scale=1) #05
            if rand_val < perc_l:
                return True
        return False


    def check_on_noise(self):
        if self.config['simulation']['sensor_noise']:
            rand_val = random.gauss(0,1)
            perc_l = scipy.stats.norm.ppf(self.config['simulation']['noise_prob'], loc=0, scale=1) #05
            if rand_val < perc_l:
                return True
        return False