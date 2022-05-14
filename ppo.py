##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   ppo.py - Implementation of the PPO algorithm           ##
##   Based on https://github.com/higgsfield/RL-Adventure-2/ ##
##############################################################
##############################################################


import logging
import numpy as np
import torch
import torch.optim as optim
from mlflow import log_metric


#################################################################################################################################

log = logging.getLogger(__name__)

############################ PROXIMAL POLICY OPTIMIZATION #######################################################################
#Class for the PPO algorithm, update loop and GAE estimation
class PPO():

    def __init__(self, cfg, model ):
        self.name = "PPO"
        self.cfg = cfg

        self.CLIP_EPSILON       = cfg['ppo']['PPO_EPSILON']
        self.PPO_EPOCHS         = cfg['ppo']['PPO_EPOCHS']
        self.CRITIC_DISCOUNT    = cfg['ppo']['CRITIC_DISCOUNT']
        self.ENTROPY_BETA       = cfg['ppo']['ENTROPY_BETA']
        self.MINI_BATCH_SIZE    = cfg['ppo']['MINI_BATCH_SIZE']
        self.GAMMA              = cfg['ppo']['GAMMA']
        self.GAE_LAMBDA         = cfg['ppo']['GAE_LAMBDA']
        self.LEARN_RATE         = cfg['ppo']['LEARN_RATE']

        self.optimizer = optim.Adam(model.parameters(), lr=self.LEARN_RATE)
        self.model = model

    #Generalized advantage estimation
    def compute_gae(self, next_value, rewards, masks, values):
        log.info('PPO GAE')
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.GAMMA * values[step + 1] * masks[step] - values[step]
            gae = delta + self.GAMMA * self.GAE_LAMBDA * masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns

    #Generates batch from an episode for the PPO update
    def ppo_iter(self, states, actions, log_probs, returns, advantage, states_img): #=None
        log.info('PPO iter')
        batch_size = states.size(0)

        #Generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // self.MINI_BATCH_SIZE):
            rand_ids = np.random.randint(0, batch_size, self.MINI_BATCH_SIZE)

            if states_img is None:
                yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], None
            else:
                yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], states_img[rand_ids, :]
            
    #Main update loop of the PPO, includes the objective function
    def ppo_update(self, states, actions, log_probs, returns, advantages, frame_idx, states_img=None):
        log.info('PPO update started')
        count_steps =     0
        sum_returns =     0.0
        sum_advantage =   0.0
        sum_loss_actor =  0.0
        sum_loss_critic = 0.0
        sum_entropy =     0.0
        sum_loss_total =  0.0

        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for i in range(self.PPO_EPOCHS):
            log.info('PPO update, epoch:%d', i)
            # grabs random mini-batches several times until we have covered all data
            for state, action, old_log_probs, return_, advantage, state_img in self.ppo_iter(states, actions, log_probs, returns, advantages, states_img):
                if state_img is not None:
                    dist, value = self.model(state, state_img, detach_cnn = True)
                else:
                    dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPSILON, 1.0 + self.CLIP_EPSILON) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = self.CRITIC_DISCOUNT * critic_loss + actor_loss - self.ENTROPY_BETA * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # PPO statistics
                sum_returns += return_.mean()
                sum_advantage += advantage.mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy

                count_steps += 1
        
        #Logging the metrics of the optimization into MlFlow
        log_metric("returns", float(sum_returns / count_steps), frame_idx)
        log_metric("advantage", float(sum_advantage / count_steps), frame_idx)
        log_metric("loss_actor", float(sum_loss_actor / count_steps), frame_idx)
        log_metric("loss_critic", float(sum_loss_critic / count_steps), frame_idx)
        log_metric("entropy", float(sum_entropy / count_steps), frame_idx)
        log_metric("loss_total", float(sum_loss_total / count_steps), frame_idx)

#################################################### END OF FILE ####################################################################