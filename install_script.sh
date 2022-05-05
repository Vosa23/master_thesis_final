#!/bin/bash

# Installing System Dependencies
# Distro check Helper
if [ -f /etc/os-release ]; then
    # freedesktop.org and systemd
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    # linuxbase.org
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    # For some versions of Debian/Ubuntu without lsb_release command
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    # Older Debian/Ubuntu/etc.
    OS=Debian
    VER=$(cat /etc/debian_version)
elif [ -f /etc/SuSe-release ]; then
    # Older SuSE/etc.
    ...
elif [ -f /etc/redhat-release ]; then
    # Older Red Hat, CentOS, etc.
    ...
else
    # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
    OS=$(uname -s)
    VER=$(uname -r)
fi

# Installing NodeJS 11.x and Dependencies
echo "${OS}"" was detected on ""${HOSTNAME}; Installing system dependencies"
if [ "${OS}" == "Ubuntu" ] ; then
  # Bare system dependencies
  sudo apt-get install -y mesa-utils libalut-dev libvorbis-dev cmake libxrender-dev libxrender1 libxrandr-dev zlib1g-dev libpng16-dev freeglut3 freeglut3-dev xvfb

  # Creating and accessing Sandbox folder for dependencies install
  if [ ! -d "dependecies" ]; then
    mkdir dependencies
  fi
  rm -rf ./dependencies/*
  cd dependencies

  PROC_COUNT="$(nproc)"
  # Installing PLIB dependecies
  echo "Installing PLIB 1.8.5"
  wget http://plib.sourceforge.net/dist/plib-1.8.5.tar.gz
  tar xvfz plib-1.8.5.tar.gz --same-owner
  cd plib-1.8.5
  export CFLAGS="-fPIC"
  export CPPFLAGS=$CFLAGS
  export CXXFLAGS=$CFLAGS
  ./configure
  make -j"${PROC_COUNT}"
  sudo make install
  export CFLAGS=
  export CPPFLAGS=
  export CXXFLAGS=
  cd ..
  rm -rf plib-1.8.5.tar.gz
  # End Installing PLIB

  # Installng OpenAL
  echo "Installing OpenAl 1.17.2"
  wget https://openal-soft.org/openal-releases/openal-soft-1.17.2.tar.bz2
  tar xvf openal-soft-1.17.2.tar.bz2 --same-owner
  cd openal-soft-1.17.2/build
  cmake ..
  make -j"${PROC_COUNT}"
  sudo make install
  cd ../../..
  rm -rf openal-soft-1.17.2.tar.bz2
fi

if [ "${OS}" == "CentOS Linux" ] ; then
  # System dependencies
  yum install -y mesa-libGL{,-devel} freealut{,-devel} libvorbis{,-devel} cmake3 libXrender{,-devel} libXrandr{,-devel} zlib{,-devel} libpng{,-devel}
fi

# Instaling Torcs itself
echo "Installing Torcs"

#If PIP not on the machine:
#sudo apt-get install python3-pip

#Python 3.6.9
echo "Install packages"

sudo pip3 install gdown

#Activating python environment and installation of libraries
python3 -m venv venv/
source ./venv/bin/activate
pip install -r requirements.txt

cd gym_torcs
echo $PWD

#Downloading of TORCS binaries
# https://drive.google.com/file/d/1WiWt5ln0D1BO_5-z0BaDXYFmqeaYuGQP/view?usp=sharing
gdown --id 1WiWt5ln0D1BO_5-z0BaDXYFmqeaYuGQP
tar -xvf vtorcs-RL-color.tar
rm vtorcs-RL-color.tar
rm -r vtorcs-RL-color/data/tracks/road/mlflow

#Installation of TORCS
cd vtorcs-RL-color
./configure

sudo make
sudo make install
sudo make datainstall
cd ../..

echo "Install complete"


#Change of TORCS screen resolution:
#/usr/local/share/games/torcs/config/screen.xml
echo "Changing TORCS screen resolution to 64x64"

sed -i "/name=\"x\"/ s/640/64/g" /usr/local/share/games/torcs/config/screen.xml
sed -i "/name=\"y\"/ s/480/64/g" /usr/local/share/games/torcs/config/screen.xml

