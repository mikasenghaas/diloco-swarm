# SWARM

This repository is a re-implementation and benchmark of [SWARM](https://arxiv.org/abs/2301.11913) parallelism. 

## Remote Setup

First, copy the `.env` file to the server and ssh into it.

```bash
USER=<user>
HOST=<host>
PORT=<port>
PERSISTENT_DIR=<persistent_dir>

scp -P $PORT -i ~/.ssh/primeintellect .env $USER@$HOST:$PERSISTENT_DIR/.env
ssh $USER@$HOST -p $PORT -i ~/.ssh/primeintellect
```

Then, clone the repository and run the setup script.

```bash
git clone https://github.com/mikasenghaas/swarm.git $PERSISTENT_DIR/swarm
mv $PERSISTENT_DIR/.env $PERSISTENT_DIR/swarm/.env
bash $PERSISTENT_DIR/swarm/scripts/setup.sh
```