#!/bin/bash

# Setup the user-specific environment.
#
# Author: Helio Perroni Filho

# Configure git and setup git-lfs.
git config --global core.editor "nano"
git lfs install

# Create a Python virtual environment to hold installed packages.
python -m venv --system-site-packages cstrack
source cstrack/bin/activate

# Set up PIP.
python -m pip install --upgrade pip

# Install Python dependencies.
# The installation is performed in steps as a workaround to
# installation failure issues with cython-bbox and onnxoptimizer.
python -m pip install -r requirements_1.txt
python -m pip install -r requirements_2.txt --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements_3.txt
