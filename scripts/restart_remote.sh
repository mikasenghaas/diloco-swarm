#!/bin/bash

# Exit on error
set -e

# Load .env
export $(grep -v '^#' .env | xargs)

# Edit path
export PATH="$PERSISTENT_DIR/.local/bin:$PERSISTENT_DIR/miniconda/bin:$PATH"

# Conda environment setup
source $PERSISTENT_DIR/miniconda/bin/activate

# Install tmux and vim
apt-get update
apt-get install -y tmux vim

echo "Done!"