#!/bin/bash

# Exit on error
set -e

# Load .env
export $(grep -v '^#' .env | xargs)

# Clone repository with strict host key checking disabled
GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' git clone git@github.com:mikasenghaas/swarm.git $PERSISTENT_DIR/swarm

# Miniconda setup
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $PERSISTENT_DIR/install_miniconda.sh
bash $PERSISTENT_DIR/install_miniconda.sh -b -p $PERSISTENT_DIR/miniconda
export PATH="$PERSISTENT_DIR/.local/bin:$PERSISTENT_DIR/miniconda/bin:$PATH"

# Conda environment setup
source $PERSISTENT_DIR/miniconda/bin/activate
conda init bash
conda create -y -n swarm python~=3.10.15
conda activate swarm

# Dependencies
PIP_FLAGS="--root-user-action ignore --upgrade pip"
pip install $PIP_FLAGS -r $PERSISTENT_DIR/swarm/requirements.txt

# W&B and HF login from .env
export $(grep -v '^#' .env | xargs)
python -m wandb login $WANDB_TOKEN
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# Copy .env to home directory
cp ~/.env $PERSISTENT_DIR

# Install tmux and vim
apt get update
apt get install -y tmux vim

echo "Done!"