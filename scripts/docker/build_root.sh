#!/bin/bash

# Setup the system-wide environment.
#
# Author: Helio Perroni Filho

# Update system and install basic utilities.
apt update && apt upgrade -y
apt install -y \
  cmake \
  curl \
  ffmpeg \
  git \
  git-lfs \
  nano \
  software-properties-common \
  sudo

# Give default user passwordless sudo permission.
echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install Python dependencies.
# python3-pyqt5 is installed as a quick way to pull in OpenCV dependencies, it
# can be removed when this environment is eventually merged with dingobot_vision.
apt install -y \
  python-is-python3 \
  python3-pip \
  python3-venv \
  python3-pyqt5
