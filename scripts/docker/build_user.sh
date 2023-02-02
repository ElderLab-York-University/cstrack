#!/bin/bash

# Setup the user-specific environment.
#
# Author: Helio Perroni Filho

# Configure git and setup git-lfs.
git config --global core.editor "nano"
git lfs install

# Create a Python virtual environment to hold installed packages.
python -m venv cstrack
source cstrack/bin/activate

# Set up PIP.
python -m pip install --upgrade pip

# Install Python dependencies.
# The installation is performed in two steps as a workaround to the
# cython-bbox installation failure issue.
python -m pip install -r requirements_1.txt
python -m pip install -r requirements_2.txt
