
##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   __init__.py - init of custom Gym environment - TORCS   ##
##                                                          ##
##############################################################
##############################################################

from gym.envs.registration import register

# Torcs Custom
register(
	id="Torcs-v0",
	entry_point='gym_torcs.torcs_env:TorcsEnv'
)
