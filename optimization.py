
##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   optimization.py - Main optimization file - Gym loop    ##
##   Based on https://github.com/higgsfield/RL-Adventure-2/ ##
##############################################################
##############################################################

import os
import sys
import copy
import logging
import numpy as np
import gym
import gym_torcs
import torch
import mlflow
from mlflow import log_metric, log_param, log_artifact, start_run
from datetime import datetime
import pickle
from cnn import ConvNetActorCritic
from nn import ActorCritic
from ppo import PPO

#################################################################################################################################

log = logging.getLogger(__name__)

#Function that logs source files into MlFlow
def log_files(cfg):
    log_artifact(cfg['setup']['cfg_name'])
    log_artifact('./gym_torcs/torcs_env.py')
    log_artifact('optimization.py')
    log_artifact('ppo.py')
    log_artifact('nn.py')
    log_artifact('cnn.py')

#Function that logs all parameters into MlFlow 
def log_params(cfg, torcs=False):
    log_param('PPO_EPSILON',            cfg['ppo']['PPO_EPSILON'])
    log_param('PPO_EPOCHS',             cfg['ppo']['PPO_EPOCHS'])
    log_param('CRITIC_DISCOUNT',        cfg['ppo']['CRITIC_DISCOUNT'])
    log_param('ENTROPY_BETA',           cfg['ppo']['ENTROPY_BETA'])
    log_param( 'MINI_BATCH_SIZE',       cfg['ppo']['MINI_BATCH_SIZE'])
    log_param('GAMMA',                  cfg['ppo']['GAMMA'])
    log_param( 'GAE_LAMBDA',            cfg['ppo']['GAE_LAMBDA'])
    log_param( 'LEARN_RATE',            cfg['ppo']['LEARN_RATE'])

    log_param('ALGORITHM',              cfg['simulation']['algorithm'])
    log_param('NUM_OF_LAYERS',          cfg['ppo']['NUM_OF_LAYERS'])
    log_param('NUM_ENVS',               cfg['ppo']['NUM_ENVS'])
    log_param('HIDDEN_SIZE_AC',         cfg['ppo']['HIDDEN_SIZE_AC'])
    log_param('HIDDEN_SIZE_CNN',        cfg['ppo']['HIDDEN_SIZE_CNN'])
    log_param('OUT_CNN',                cfg['ppo']['OUT_CNN'])
    log_param('PPO_STEPS',              cfg['ppo']['PPO_STEPS'])
    log_param('TOTAL_EPOCHS',           cfg['ppo']['TOTAL_EPOCHS'])
    log_param('TARGET_REWARD',          cfg['ppo']['TARGET_REWARD'])
    log_param('EARLY_STOP',             cfg['ppo']['EARLY_STOP'])
    log_param('TEST_STEPS',             cfg['ppo']['TEST_STEPS'])


    if torcs:
        log_param('MODE',               cfg['simulation']['mode'])
        log_param('device',             cfg['simulation']['DEVICE'])
        log_param('TRACK',              cfg['setup']['track'])
        log_param('TRACK_LENGTH',       cfg['setup']['track_property']['length'])
        log_param('TRACK_INFO',         cfg['setup']['track_property'])
        log_param('TEST_TRACK',         cfg['setup']['test_track'])
        log_param('TEST_DETERM_ACTIONS',cfg['setup']['determ_actions_test'])

        log_param('vision',             cfg['setup']['vision'])
        log_param('race_speed',         cfg['setup']['race_speed'])
        log_param('reward_function',    cfg['setup']['reward_function'])
        log_param('rendering',          cfg['setup']['rendering'])

        log_param('normalize',          cfg['setup']['normalize'])
        log_param('IMG_MODE',           cfg['ppo']['IMG_MODE'])

        log_param('comm_failure',       cfg['simulation']['comm_failure'])
        log_param('comm_delay',         cfg['simulation']['comm_delay'])
        log_param('sensor_noise',       cfg['simulation']['sensor_noise'])

        log_param('architecture',         cfg['simulation']['architecture'])

        log_param('record_samples',     cfg['simulation']['record_samples'])
        log_param('cnn_to_sensors',     cfg['simulation']['cnn_to_sensors'])
        log_param('cnn_to_sensors_path',cfg['simulation']['cnn_to_sensors_path'])
        log_param('load_model',         cfg['simulation']['load_model'])
        log_param('CNN_TYPE',           cfg['simulation']['cnn_type'])
        log_param('IMG_TYPE',           cfg['simulation']['img_type'])
        log_param('IMG_NUM',            cfg['simulation']['img_num'])


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

############################ TESTING THE ENVIRONMENT #######################################################################
#Function used for testing and evaluation of the agent
def test_env_torcs( env, model, device, obs_vars, TEST_STEPS, track_length, model_cnn=None, CNN_TO_SENSORS=False,
                    RECORD_SAMPLES=False, deterministic=False, RECORD_SAMPLES_PATH='record_samples_default.pth', eval=False):
    
    if isinstance(model, ConvNetActorCritic)  or CNN_TO_SENSORS or RECORD_SAMPLES:
        state, state_img = env.reset(relaunch=True)
    else:
        state = env.reset(relaunch=True)

    done =      False
    measured =  False
    total_reward =  0
    frame_idx =     0
    
    if RECORD_SAMPLES:
        imgs = []
        sensors = []

    if eval:
        mlflow.set_tag('eval', 'yes')
    
    while not done and frame_idx < TEST_STEPS:
        if frame_idx == 0:
            time_start = datetime.now()

        distRaced = state.pop('distRaced', None)   #bcs of distRaced is not part of the sensors for the agent, just for internal statistics
        damage = state.pop('damage', None)
        speedX = state['speedX']*300
        
        print('############# TESTING STEP ################')
        print('TEST frame:: ', frame_idx)
        print('step:: ', frame_idx )
        print('distRaced:: ', distRaced)    #previous state
        print('speedX:: ', speedX)          #previous state

        percent_of_track = np.round( ((distRaced/track_length) * 100), 2)

        if percent_of_track > 100 and not measured:
            measured = True
            time_end = datetime.now()
            lap_time = time_end - time_start
            print('TEST_lap_time:: ', lap_time.total_seconds())
            log_metric( 'TEST_lap_time', float(lap_time.total_seconds()))

        #ADD/REMOVE SENSORS YOU WANT
        if RECORD_SAMPLES:
            state_pickle = obs_preprocess_fn(state, ['angle','track','trackPos'])              #['angle','track']
            state_pickle = torch.FloatTensor(state_pickle).to(device)

        state = obs_preprocess_fn(state, obs_vars)
        state = torch.FloatTensor(state).to(device)

        if isinstance(model, ConvNetActorCritic):
            state_img = torch.FloatTensor(state_img).to(device)

            #save samples
            if RECORD_SAMPLES:
                imgs.append(state_img)
                sensors.append(state_pickle)
            dist, _ = model(state, state_img)
        elif CNN_TO_SENSORS:
            state_img = torch.FloatTensor(state_img).to(device)
            cnn_state = model_cnn(state_img)
            state = torch.hstack( (state, cnn_state) )
            dist, _ = model(state)
        else:
            if RECORD_SAMPLES:      #from sensor only learned agent
                state_img = torch.FloatTensor(state_img).to(device)
                imgs.append(state_img)
                sensors.append(state_pickle)
            dist, _ = model(state)

        if deterministic:
            action = dist.mean.detach().cpu().numpy()
        else:
            action = dist.sample().cpu().numpy()
        
        print('Action in testing mode:: ', action)
        action = np.squeeze(action)

        if isinstance(model, ConvNetActorCritic) or CNN_TO_SENSORS or RECORD_SAMPLES:
            (next_state, next_state_img), reward, done, _ = env.step(action)
        else:
            next_state, reward, done, _ = env.step(action)

        if not done:
            if isinstance(model, ConvNetActorCritic) or CNN_TO_SENSORS or RECORD_SAMPLES:
                state_img = next_state_img
            state = next_state
            total_reward += reward
            frame_idx += 1

        print('reward:: ', reward)

    if percent_of_track > 100:
        mlflow.set_tag('TEST_one_lap_driven', 'yes')

    log_metric('TEST_distRaced',        float(distRaced) )
    log_metric('TEST_percent_of_track', float(percent_of_track) )
    log_metric('TEST_total_reward',     float(total_reward) )
    print('total_reward:: ', total_reward)

    #Pickle samples
    if RECORD_SAMPLES:
        samples = dict()
        samples['data'] = imgs
        samples['labels'] = sensors
        
        with open(RECORD_SAMPLES_PATH, 'ab') as fn: #appending
            pickle.dump(samples, fn, pickle.HIGHEST_PROTOCOL)
        print('PICKLES SAVED...')

    return total_reward

#Function that preprocesses the sensory data for the Actor-critic network
def obs_preprocess_fn( dict_obs, obs_vars ):

    if not isinstance(dict_obs, dict):
        return dict_obs

    vars_ = [v for (k,v) in dict_obs.items() if k in obs_vars]
    
    v = np.hstack([v for v in vars_])
    v = v[np.newaxis,:]
    return v


############################ SIMULATION CONTROL #######################################################################

#Main function of the optimization algorithm
def run(cfg, obs_vars, obs_preprocess):

    #Command line has to be emptied for correct startup of TORCS
    sys.argv = [sys.argv[0]]

    #Loading the hyperparameters from config file
    ppo_version         = cfg['simulation']['algorithm']
    track_dict          = cfg['setup']['track_property']
    track_dict['name']  = cfg['setup']['track']
    track_length        = track_dict['length']

    DETERM_ACTIONS      = cfg['setup']['determ_actions_test']

    RECORD_SAMPLES      = cfg['simulation']['record_samples']
    RECORD_SAMPLES_PATH = cfg['simulation']['record_samples_path']
    NUM_TESTS           = cfg['ppo']['NUM_TESTS']
    NUM_ENVS            = cfg['ppo']['NUM_ENVS']
    NUM_LAYERS          = cfg['ppo']['NUM_OF_LAYERS']
    STD_DEV             = cfg['ppo']['STD_DEV']
    HIDDEN_SIZE_AC      = cfg['ppo']['HIDDEN_SIZE_AC']
    HIDDEN_SIZE_CNN     = cfg['ppo']['HIDDEN_SIZE_CNN']
    OUT_CNN             = cfg['ppo']['OUT_CNN']

    TARGET_REWARD       = cfg['ppo']['TARGET_REWARD']
    PPO_STEPS           = cfg['ppo']['PPO_STEPS']
    EARLY_STOP          = cfg['ppo']['EARLY_STOP']
    TOTAL_EPOCHS        = cfg['ppo']['TOTAL_EPOCHS']
    TEST_STEPS          = cfg['ppo']['TEST_STEPS']
    SAVE_EPOCHS         = cfg['ppo']['SAVE_EPOCHS']
    TEST_EPOCHS         = cfg['ppo']['TEST_EPOCHS']
    TARGET_STEP         = cfg['ppo']['TARGET_STEP']

    CNN_TO_SENSORS      = cfg['simulation']['cnn_to_sensors']
    CNN_TO_SENSORS_PATH = cfg['simulation']['cnn_to_sensors_path']
    CNN_TYPE            = cfg['simulation']['cnn_type']

    #Start of logging into MLFlow
    mlflow.set_experiment('ppo_torcs_CNN-archs')
    experiment = mlflow.get_experiment_by_name('ppo_torcs_CNN-archs')
    
    now = datetime.now()
    now_time = now.strftime("%m%d%Y-%H-%M-%S")
    start_run(experiment_id=experiment.experiment_id ,run_name= ppo_version + '_' + now_time )
    mlflow.set_tag('version', '0.1')

    log_params(cfg, True)
    log_files(cfg)

    #Used for parallelization of environment
    def make_env(rank):
        def _thunk():
            env = gym.make( "Torcs-v0", vision=cfg['setup']['vision'], rendering=cfg['setup']['rendering'],
                throttle=cfg['setup']['throttle'], race_speed=cfg['setup']['race_speed'],
                obs_vars=obs_vars, obs_normalization=cfg['setup']['normalize'], rank=rank,
                reward_function_used=cfg['setup']['reward_function'], obs_preprocess_fn=obs_preprocess, track_dict=track_dict
                ,config=cfg)
            return env
        return _thunk

    
    # Autodetect CUDA
    cuda_avail = torch.cuda.is_available()
    if cuda_avail and cfg['simulation']['DEVICE'] == "cuda":
        device   = torch.device("cuda")
    else:
        device   = torch.device("cpu")
    print('Device:', device)


    # Prepare environments
    if cfg['ppo']['NUM_ENVS'] > 1:
        pass
        envs = [make_env(i) for i in range(NUM_ENVS)]
        envs = SubprocVecEnv(envs)  #Not further tested
        # env = gym.make(ENV_ID)
    else:
        envs = gym.make( "Torcs-v0", vision=cfg['setup']['vision'], rendering=cfg['setup']['rendering'],
                throttle=cfg['setup']['throttle'], race_speed=cfg['setup']['race_speed'],
                obs_vars=obs_vars, reward_function_used=cfg['setup']['reward_function'],
                obs_preprocess_fn=obs_preprocess, obs_normalization=cfg['setup']['normalize'], track_dict=track_dict,
                config=cfg)

    #distRaced, damage removed..as it is always on, so also during CNN_TO_SENSORS
    num_inputs  = envs.observation_space.shape[0] - 2
    num_outputs = envs.action_space.shape[0]
    print('num_inputs:: ', num_inputs)
    print('num_outputs:: ', num_outputs)

    #ADD/REMOVE SENSORS YOU WANT
    #angle=1, track=19, trackPos=1      #used when RECORD_SAMPLES
    if CNN_TO_SENSORS:
        num_inputs += 21 #19, 20, 21


    #Loading trained model
    model_path  = cfg['simulation']['load_model']
    mode        = cfg['simulation']['mode']

    #when we eval model, which was trained with img->sensors, we need to load the CNN_TO_SENSORS as well
    if CNN_TO_SENSORS and CNN_TO_SENSORS_PATH is not None:
        model_cnn = torch.load(CNN_TO_SENSORS_PATH)
        model_cnn.to(device)
        model_cnn.eval()
        for p in model_cnn.parameters():
            p.requires_grad = False
    else:
        model_cnn = None

    if model_path is not None:
        model = torch.load(model_path)
        print('Model was loaded:: ', model_path )

        if mode == 'train':
            print('Training continues (fine tuning).')
            model.train()
        elif mode == 'eval':
            print('Evaluation of a model starts.')
            model.eval()

            test_rewards = []
            for _ in range(NUM_TESTS):
                test_reward = test_env_torcs(envs, model, device, obs_vars, TEST_STEPS, track_length, model_cnn, CNN_TO_SENSORS, RECORD_SAMPLES,
                                             deterministic=DETERM_ACTIONS, RECORD_SAMPLES_PATH=RECORD_SAMPLES_PATH, eval=True)
                test_rewards.append(test_reward)

            test_mean = np.mean(test_rewards)
            print('test_mean:: ', test_mean)
            log_metric('TEST_mean_reward', float(test_mean))
            envs.end()
            return

    elif cfg['simulation']['architecture'] == 'cnn':
        model = ConvNetActorCritic(cfg['simulation']['img_num'], num_inputs, num_outputs, HIDDEN_SIZE_AC, HIDDEN_SIZE_CNN, OUT_CNN, CNN_TYPE, STD_DEV)
    else:
        model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE_AC, NUM_LAYERS, STD_DEV).to(device)

    print(model)


    #initiating the PPO agent
    agent = PPO(cfg, model)

    frame_idx       = 0
    total_frame_idx = 0
    train_epochs    = 0
    best_reward     = None
    early_stop      = False


    #currently TARGET_REWARD or TOTAL_EPOCHS achieved
    while not early_stop:

        measured     = False
        episode_len  = 0
        avg_speed    = 0.
        damage_val   = 0.
        damage_count = 0
        total_reward = 0.
        log_probs    = []
        values       = []
        states       = []
        actions      = []
        rewards      = []
        masks        = []

        if isinstance(model, ConvNetActorCritic) or CNN_TO_SENSORS:
            states_img = []

        # Restart of TORCS because of the memory leak error
        if np.mod(train_epochs, 1) == 0:        #used to be 3, but this is better since no TIMEOUT-RECONNECT 5s countdown appears 
            if isinstance(model, ConvNetActorCritic) or CNN_TO_SENSORS:
                state, state_img = envs.reset(relaunch=True)
            else:
                state = envs.reset(relaunch=True)
        else:
            if isinstance(model, ConvNetActorCritic) or CNN_TO_SENSORS:
                state, state_img = envs.reset()
            else:
                state = envs.reset()


        #TORCS episode
        for step in range(PPO_STEPS):

            if step == 0:
                time_start = datetime.now()

            if cfg['ppo']['NUM_ENVS'] > 1:
                pass
                for st in state:
                    damage = st.pop('damage', None)
                    dist_raced = st.pop('distRaced', None)
                    speedX = st['speedX']*300
            else:
                damage = state.pop('damage', None)
                dist_raced = state.pop('distRaced', None)
                speedX = state['speedX']*300

            #Logging of each time-step into the terminal
            print('############# NEW STEP ################')
            print('total_frame_idx:: ', total_frame_idx)
            print('frame:: ',           frame_idx)
            print('step:: ',            step)
            print('distRaced:: ',       dist_raced)     #previous state
            print('speedX:: ',          speedX)         #previous state

            #Preprocess of sensor values & action sampling
            state = obs_preprocess_fn(state, obs_vars)
            state = torch.FloatTensor(state).to(device)

            if isinstance(model, ConvNetActorCritic):
                state_img = torch.FloatTensor(state_img).to(device)
                dist, value = model(state, state_img)
            elif CNN_TO_SENSORS:
                state_img = torch.FloatTensor(state_img).to(device)
                cnn_state = model_cnn(state_img)
                state = torch.hstack( (state, cnn_state) )
                dist, value = model(state)
            else:
                dist, value = model(state)

            action_for_env = dist.sample()
            # action_for_env = torch.clamp(action_for_env, min=-1., max=1.)


            action_for_alg = action_for_env
            action_for_env = np.squeeze(action_for_env)
            print('Action:: ', action_for_env)

            #Step within environment
            if isinstance(model, ConvNetActorCritic) or CNN_TO_SENSORS:
                (next_state, next_state_img), reward, done, _ = envs.step(action_for_env.cpu().numpy())
            else:
                next_state, reward, done, _ = envs.step(action_for_env.cpu().numpy())

            log_prob = dist.log_prob(action_for_env)

            #Saving the data for PPO update
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor( reward ).to(device)) #.unsqueeze(1)
            masks.append(torch.FloatTensor( np.array(1. - done) ).to(device)) #round or np.int8
            states.append(state)
            actions.append(action_for_alg)

            if isinstance(model, ConvNetActorCritic): #or CNN_TO_SENSORS:
                states_img.append(state_img)
            
            
            curr_percent_of_track = np.round( ((dist_raced/track_length) * 100), 2)
            if curr_percent_of_track > 100 and not measured:
                measured = True
                time_end = datetime.now()
                lap_time = time_end - time_start
                print('lap_time:: ', lap_time.total_seconds())
                log_metric( 'lap_time', float(lap_time.total_seconds()), frame_idx)


            print('reward:: ', reward)

            #STATS
            if not done:
                avg_speed += speedX
                episode_len += 1
                total_reward += reward

                pre_damage = damage
                damage = next_state['damage']
                if damage - pre_damage > 0:
                    damage_count += 1
                    damage_val += damage

                frame_idx += 1
            total_frame_idx += 1

            state = next_state
            if isinstance(model, ConvNetActorCritic) or CNN_TO_SENSORS:
                state_img = next_state_img

            if done:
                break
        

        #also every other sensor, has to be deleted like that + in torcs_env set to not be normalized... + (obs_vars - 1) during creation of input_num_nn
        dist_raced = next_state.pop('distRaced', None)
        damage = next_state.pop('damage', None)


        average_speed = np.round( ( avg_speed / np.float(episode_len) ), 2)
        percent_of_track = np.round( ((dist_raced/track_length) * 100), 2)
        time_elapsed = datetime.now() - now

        if percent_of_track > 100:
            mlflow.set_tag('one_lap_driven', 'yes')
            # early_stop = True
        
        #Logging of metrics into the MlFlow
        log_metric("total_reward",      float(total_reward),                    frame_idx)
        log_metric("percent_of_track",  float(percent_of_track),                frame_idx)
        log_metric("dist_raced",        float(dist_raced),                      frame_idx)
        log_metric("average_speed",     float(average_speed),                   frame_idx)
        log_metric("damage_value",      float(damage_val),                      frame_idx)
        log_metric("damage_count",      float(damage_count),                    frame_idx)
        log_metric("real_len_episode",  float(episode_len),                     frame_idx)
        log_metric("train_epochs",      float(train_epochs),                    frame_idx)
        log_metric("time_elapsed",      float(time_elapsed.total_seconds()),    frame_idx)
        log_metric("total_frame_idx",   float(total_frame_idx),                 frame_idx)

        #Printing the metrics values into terminal
        print('############# EPOCH SUMMARY ################')
        print('epoch:: ',               train_epochs)
        print('total_frame_idx:: ',     total_frame_idx)
        print('frame:: ',               frame_idx)
        print('real_len_episode:: ',    episode_len )
        print('total_reward:: ',        total_reward )
        print('distRaced:: ',           dist_raced)
        print('average_speed:: ',       average_speed)
        print('percent_of_track:: ',    percent_of_track)
        print('damage_count:: ',        damage_count)
        print('damage_value:: ',        damage_val)
        print('time_elapsed:: ',        time_elapsed)
        print('############# END OF EPOCH SUMMARY ################')

        #GAE needs one more state sample, after the end of an episode
        next_state = obs_preprocess_fn(next_state, obs_vars)
        next_state = torch.FloatTensor(next_state).to(device)

        if isinstance(model, ConvNetActorCritic):
            next_state_img = torch.FloatTensor(next_state_img).to(device)
            _, next_value = model(next_state, next_state_img)
        elif CNN_TO_SENSORS:
            next_state_img = torch.FloatTensor(next_state_img).to(device)
            next_cnn_state = model_cnn(next_state_img)
            next_state = torch.hstack( (next_state, next_cnn_state) )
            _, next_value = model(next_state)
        else:
            _, next_value = model(next_state)

        #Calculation of Generalized Advantage Estimaton (GAE) - Lambda return
        returns = agent.compute_gae(next_value, rewards, masks, values)

        #Preparation of data for PPO update
        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        advantage = normalize(advantage)
        
        if isinstance(model, ConvNetActorCritic): #or CNN_TO_SENSORS:
            states_img = torch.cat(states_img)


##################################################################################################


        #We update the PPO based on returns calculated above
        if isinstance(model, ConvNetActorCritic):
            agent.ppo_update(states, actions, log_probs, returns, advantage, frame_idx, states_img=states_img)
        else:
            #here should be already states - including sensors from cnn, I suppose
            agent.ppo_update(states, actions, log_probs, returns, advantage, frame_idx)
        train_epochs += 1


        #every 10 epochs - testing
        if train_epochs % TEST_EPOCHS == 0:
            print('TESTING OF TORCS ENV, episode:: ',train_epochs )
            
            if cfg['setup']['test_track'] is not None:
                test_track_dict = copy.deepcopy(track_dict)
                test_track_dict['name']  = cfg['setup']['test_track']
            else:
                test_track_dict = track_dict
            
            #Creating new Gym environment for the evaluation/testing 
            env = gym.make( "Torcs-v0", vision=cfg['setup']['vision'], rendering=cfg['setup']['rendering'],
                                throttle=cfg['setup']['throttle'], race_speed=cfg['setup']['race_speed'],
                                obs_vars=obs_vars,reward_function_used=cfg['setup']['reward_function'],
                                obs_normalization=cfg['setup']['normalize'], obs_preprocess_fn=obs_preprocess, track_dict=test_track_dict,
                                rank=42, config=cfg)

            test_reward = test_env_torcs(env, model, device, obs_vars, TEST_STEPS, track_length, model_cnn, CNN_TO_SENSORS, deterministic=DETERM_ACTIONS)
            env.end()


            print('Frame %s. reward: %s' % (frame_idx, test_reward))
            log_metric("TEST_reward", float(test_reward), frame_idx)
            
            log_time = datetime.now()
            log_time_name = log_time.strftime("%m%d%Y-%H-%M-%S_")

            #logging the model  #Create it as a parameter
            if train_epochs % SAVE_EPOCHS == 0:
                log_name = "_log_%+.3f_%d.pth" % ( test_reward, frame_idx)
                mlflow.pytorch.log_model(model, log_time_name + str(time_elapsed.total_seconds()) + log_name)

            if best_reward is not None:
                log_metric("BEST_reward", float(best_reward), frame_idx) #maybe None error
                

            # Save a checkpoint every time we achieve a best reward
            if best_reward is None or best_reward < test_reward:
                best_reward = test_reward

                print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                name = "_best_%+.3f_%d.pth" % ( test_reward, frame_idx)
                
                #Saving here is not necessary, as we save it inside the MlFlow - so it is better organized (model at the same place as metrics for given exp)
                # fname = os.path.join('.', 'checkpoints', log_time_name + str(time_elapsed.total_seconds()) + name)
                # torch.save(model, fname)
                # mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path=name)

            ########################################################

            if EARLY_STOP:
                if test_reward > TARGET_REWARD: early_stop = True
            else:
                if train_epochs > TOTAL_EPOCHS : early_stop = True
        
        if EARLY_STOP:
            if frame_idx > TARGET_STEP: early_stop = True


    #Saving Last Model - before the end of optimization
    log_time_end = datetime.now()
    log_time_end_name = log_time_end.strftime("%m%d%Y-%H-%M-%S_")
    log_name_end = "_log_%+.3f_%d.pth" % ( test_reward, frame_idx)
    mlflow.pytorch.log_model(model, log_time_end_name + str(time_elapsed.total_seconds()) + log_name_end)

    #Closing the environment
    envs.end()


################################ END OF FILE ####################################