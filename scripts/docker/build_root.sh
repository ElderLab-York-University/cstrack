#!/bin/bash

# Setup the system-wide environment.
#
# Author: Helio Perroni Filho

# Update system and install basic utilities.
apt update && apt upgrade -y
apt install -y \
  ffmpeg \
  git \
  git-lfs \
  nano \
  sudo

# Give default user passwordless sudo permission.
echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install Python dependencies.
apt install -y \
  libnvinfer-bin \
  python-is-python3 \
  python3-libnvinfer-dev \
  python3-pip \
  python3-pycuda \
  python3-venv \
  python3-pyqt5 \
  tensorrt-dev
