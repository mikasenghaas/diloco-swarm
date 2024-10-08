#!/bin/bash

# Exit on error
set -e

# Clone repository with strict host key checking disabled
GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' git clone git@github.com:mikasenghaas/swarm.git $HOME/swarm

# Miniconda setup
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/install_miniconda.sh
bash $HOME/install_miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/.local/bin:$HOME/miniconda/bin:$PATH"

# Conda environment setup
source $HOME/miniconda/bin/activate
conda init bash
conda create -y -n swarm python~=3.10.15
conda activate swarm

# Dependencies
PIP_FLAGS="--root-user-action ignore --upgrade pip"
pip install $PIP_FLAGS -r $HOME/swarm/requirements.txt

# W&B and HF login from .env
export $(grep -v '^#' .env | xargs)
python -m wandb login $WANDB_TOKEN
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

echo "Setup complete. You can now use 'conda activate swarm' to activate the environment."