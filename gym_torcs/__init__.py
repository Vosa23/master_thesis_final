#DVO initiation of custom Gym environment for TORCS

from gym.envs.registration import register

# Torcs Custom
register(
	id="Torcs-v0",
	entry_point='gym_torcs.torcs_env:TorcsEnv'
)
