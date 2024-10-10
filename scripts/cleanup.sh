#!/bin/bash

# Check for required arguments
if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
    echo "Usage: $0 <USER> <HOST> <PERSISTENT_DIR> [<PORT>]"
    exit 1
fi

USER=$1
HOST=$2
PERSISTENT_DIR=$3
if [ "$#" -eq 4 ]; then
    PORT=$4
else
    PORT=22
fi

# Set up SSH command
SSH_FLAGS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=QUIET"
SSH_CMD="ssh $SSH_FLAGS $USER@$HOST -p $PORT -i ~/.ssh/prime"

# Execute remote cleanup
echo "Cleaning up remote server..."
$SSH_CMD << EOF > /dev/null 2>&1
    set -e
    bash ~/cleanup_remote.sh
EOF

echo "Done!"

# Connect to instance
CMD="ssh $USER@$HOST -p $PORT"
echo "Connect to Prime instance using \`$CMD\` (Copied to clipboard!)" && echo $CMD | pbcopy