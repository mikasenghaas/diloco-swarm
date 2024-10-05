# SWARM

This repository is a re-implementation and benchmark of [SWARM](https://arxiv.org/abs/2301.11913) parallelism. 

## Remote Setup

```bash
PERSISTENT_DIR=/workspace
git clone https://github.com/mikasenghaas/swarm.git $PERSISTENT_DIR/swarm
scp .env <user>@<host>:$PERSISTENT_DIR/swarm/.env
bash $PERSISTENT_DIR/swarm/scripts/setup.sh
```