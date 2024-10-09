#!/bin/bash

# Check for required arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <SSH_STRING> <PERSISTENT_DIR>"
    exit 1
fi

# Check that .env, .gitconfig and .sshconfig exist
if [ ! -f .env ] || [ ! -f .gitconfig ] || [ ! -f .sshconfig ]; then
    echo "Error: .env, .gitconfig or .sshconfig does not exist"
    exit 1
fi

# Parse user, host and port from the input string
SSH_STRING="$1"
PERSISTENT_DIR="$2"
USER=$(echo "$SSH_STRING" | awk '{print $2}' | cut -d'@' -f1)
HOST=$(echo "$SSH_STRING" | awk '{print $2}' | cut -d'@' -f2)
PORT=$(echo "$SSH_STRING" | awk '{print $4}')

# Add or update PERSISTENT_DIR in .env
sed -i '' '/^PERSISTENT_DIR=/d' .env
echo "PERSISTENT_DIR=$PERSISTENT_DIR" >> .env

# Set up SSH and SCP commands
SSH_FLAGS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=QUIET "
SSH_CMD="ssh $SSH_FLAGS $USER@$HOST -p $PORT -i ~/.ssh/prime"
SCP_CMD="scp $SSH_FLAGS -P $PORT -i ~/.ssh/prime"

# Transfer files
echo "Transferring files..."
$SCP_CMD .env $USER@$HOST:~
$SCP_CMD .gitconfig $USER@$HOST:~/.gitconfig
$SCP_CMD .sshconfig $USER@$HOST:~/.ssh/config
$SCP_CMD scripts/setup_remote.sh $USER@$HOST:~
$SCP_CMD scripts/cleanup_remote.sh $USER@$HOST:~
$SCP_CMD ~/.ssh/github-personal $USER@$HOST:~/.ssh/github-personal

# Execute remote setup
echo "Setting up remote server..."
$SSH_CMD << EOF > /dev/null 2>&1
    set -e
    bash ~/setup_remote.sh
EOF
echo "Done!"

# Connect to instance
CMD="ssh $USER@$HOST -p $PORT"
echo "Connect to Prime instance using \`$CMD\` (Copied to clipboard!)" && echo $CMD | pbcopy