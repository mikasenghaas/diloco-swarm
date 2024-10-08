#!/bin/bash

# Check for required arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 \"ssh <user>@<host> -p <port> -i <private_key>\""
    exit 1
fi

# Parse user, host and port from the input string
SSH_STRING="$1"
USER=$(echo "$SSH_STRING" | awk '{print $2}' | cut -d'@' -f1)
HOST=$(echo "$SSH_STRING" | awk '{print $2}' | cut -d'@' -f2)
PORT=$(echo "$SSH_STRING" | awk '{print $4}')

# Set up SSH command
SSH_FLAGS="-o StrictHostKeyChecking=no -o LogLevel=QUIET -q"
SSH_CMD="ssh $SSH_FLAGS $USER@$HOST -p $PORT -i ~/.ssh/prime"

# Execute remote cleanup
echo "Cleaning up remote server..."
$SSH_CMD << EOF > /dev/null 2>&1
    set -e
    bash ~/cleanup_remote.sh
EOF

echo "Done!"