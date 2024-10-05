#!/bin/bash

# Load .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Install miniconda
set -e
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $PWD/miniconda
export PATH="$PWD/miniconda/bin:$PATH"
conda init
source ~/.bashrc

# Create conda environment
conda create -n swarm python~=3.10.15 -y
conda activate swarm

# Install additional dependencies
cd swarm
pip install --user --upgrade pip
pip install --user --no-cache-dir -r requirements.txt

# Setup git
echo "Setting up git"
git config --global user.name "mikasenghaas"
git config --global user.email "mikasenghaas@gmail.com"
git config --global credential.helper store
echo "https://mikasenghaas:${GIT_TOKEN}@github.com" > ~/.git-credentials

# Setup keys
echo "Setting up W&B API key"
python -m wandb login $WANDB_TOKEN

# Setup W&B
echo "Setting up HF login"
huggingface-cli login --token $HF_TOKEN --add-to-git-credential