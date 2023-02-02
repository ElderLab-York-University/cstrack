#!/bin/bash

# Install host-side dependencies.
#
# Author: Helio Perroni Filho

# Show commands as they're executed, exit on error.
set -x -e

# Update installed packages.
apt update
apt upgrade -y

# Install basic utilities.
apt install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  git \
  git-lfs \
  gnupg-agent \
  screen \
  software-properties-common

# Setup access to the Docker package repository.
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
apt-key fingerprint 0EBFCD88
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Setup access to the NVIDIA container toolkit repository.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the system package index.
apt update

# Install Docker and NVIDIA extensions.
apt install -y docker-ce docker-ce-cli containerd.io nvidia-docker2

# Add the current user to the docker group.
usermod -aG docker $USER

echo "Reboot to bring changes into effect."
