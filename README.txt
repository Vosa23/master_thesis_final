
##############################################################
##############################################################
##   Bc. David Vosol (xvosol00)                             ##
##   VUT FIT 2021/2022                                      ##
##   Master's Thesis implementation                         ##
##   README                                                 ##
##                                                          ##
##############################################################
##############################################################

#Easy Installation:

Run the automated installation script:

sudo ./install_script.sh

This should automate the complete installation process, which includes these steps:

This should first download the TORCS binaries, then install the neccesary libraries for it.

As a next thing it should activate the python environment and install the python libraries through requirements.txt file

Next it should change the screen resolution of the TORCS to 64x64 (if it is not, read the guide below)

Unfortunatelly this is the only way we are able to change the TORCS screen resolution. It is needed for the experiments that include camera sensor
Also camera can be used only when we are directly on the Linux machine, unfortunatelly it does not work through xserver (X11) forwarding.
(The agent then receives only black image and not the real in-game screen image)

################################################################

#How to run the program:

	source ./venv/bin/activate
	python3 run.py -c config.yaml

################################################################

#Complete Installation: (in case of problems, Ubuntu version)

These common libraries should be installed before hand: 

mesa-utils libalut-dev libvorbis-dev cmake libxrender-dev libxrender1 libxrandr-dev zlib1g-dev libpng16-dev freeglut3 freeglut3-dev xvfb

Just in case that the TORCS installation was not successfull: the TORCS depends on the plib-1.8.5 and openal-soft-1.17.2 library
Its installation procedure can be followed in the install_script.sh file.


#Manual download of TORCS binaries:
Download from link: https://drive.google.com/file/d/1WiWt5ln0D1BO_5-z0BaDXYFmqeaYuGQP/view?usp=sharing

Or use gdown utility:
	sudo pip install gdown
	cd gym_torcs
	gdown --id 1WiWt5ln0D1BO_5-z0BaDXYFmqeaYuGQP
	tar -xvf vtorcs-RL-color.tar


#Manual installation of TORCS:

	cd vtorcs-RL-color
	sudo ./configure
	sudo make
	sudo make install
	sudo make datainstall


#Manual activation of python environment:

	source ./venv/bin/activate  


#Manual installation of python dependencies:

	pip install -r requirements.txt


#Change of TORCS screen resolution:

If for any reason the TORCS screen resolution is not changed automatically from 640x480 to 64x64:

The file should be located at: /usr/local/share/games/torcs/config/screen.xml and the x,y values should be changed to 64


#Change to correct camera view

Also if the camera screen is not from first person view (but somehow angled and the view does not appear that it is from the car) 
press F2 to change the in-game camera view (probably multiple times, to get the cam view with greatest field of view).


#Kill zombied TORCS processes

If the simulation ends or is exited by ctrl+C and the TORCS game window stays open, kill TORCS processess by:

	ps aux | grep 'torcs' | awk '{print $2}' | xargs kill

It is also a good practice from time to time run this command too. So the TORCS processess do not occupy the memory.


#MlFlow UI
To run the MlFlow web service, it is required to run the command in separate terminal window:

	mlflow ui

################################################################


#Config file:

The file is config.yaml located in root directory
All experiment related settings are located there.
Also the speed of simulation, the track selection, loading of learned model, etc.
To every option there is a comment, explaining what each parameter does


#Implementation files:

run.py				- from where the implementation is started
optimization.py			- the main optimization loop
nn.py				- implementation of feed forward neural networks
cnn.py				- implementation of convolutional neural networks
ppo.py				- implementation of the PPO algorithm
ppo_cnn.py			- for older learned agent models, all previous files merged into one
config.yaml			- main configuration file for experiment setup
tracks.yaml			- configuration of available race tracks
CNN_model_for_sensors.ipynb	- python notebook for CNN training (Hybrid architecture - camera to sensors prediction)
gym_torcs/__init__		- necessary file for registration of the custom Gym TORCS environment
gym_torcs/snakeoil3_gym.py	- client for UDP communication with the TORCS server
gym_torcs/torcs_env.py		- Gym wrapper over the SnakeOil client

#Race track configs
raceconfigs/default.xml
raceconfigs/eroad.xml
raceconfigs/e-track-1.xml
raceconfigs/e-track-2.xml
raceconfigs/e-track-3.xml
raceconfigs/e-track-4.xml
raceconfigs/e-track-6.xml
raceconfigs/forza.xml
raceconfigs/g-track-2.xml
raceconfigs/g-track-3.xml
raceconfigs/michigan.xml

#TORCS server files:
gym_torcs/vtorcs-RL-color	- implementation and TORCS binaries
checkpoints/			- trained agent models are saved there
mlruns/				- statistics from individual runs are saved there


