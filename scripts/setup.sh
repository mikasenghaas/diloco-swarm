#!/bin/bash

# Install miniconda
set -e
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Clone repository
git clone https://github.com/mikasenghaas/swarm.git
cd swarm

# Create conda environment
conda create -n swarm python~=3.10.15 pip
conda activate swarm

# Install additional dependencies
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt