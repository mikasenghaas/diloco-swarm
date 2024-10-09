#!/bin/bash

# Exit on error
set -e

# Load .env
export $(grep -v '^#' .env | xargs)

# Cleanup
rm ~/.env ~/setup_remote.sh ~/cleanup_remote.sh ~/.gitconfig ~/.config/ssh/config ~/.ssh/github-personal
rm -rf $PERSISTENT_DIR/miniconda $PERSISTENT_DIR/swarm $PERSISTENT_DIR/install_miniconda.sh
