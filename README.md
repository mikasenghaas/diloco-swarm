# SWARM

This repository is a re-implementation and benchmark of [SWARM](https://arxiv.org/abs/2301.11913) parallelism.

## ⚙️ Setup

To set up the project on a fresh [Prime Compute Instance](https://www.app.primeintellect.com/), follow these steps:

**1. Clone the Repository**

```bash
gh repo clone mikasenghaas/swarm && cd swarm
```

**2. Configure Environment**

Configure environment variables for W&B, HF, Git, and SSH. To do this, first copy the example files:

```bash
cp .env.example .env
cp .gitconfig.example .gitconfig
cp .sshconfig.example .sshconfig
```

Then, edit the files by filling your `WANDB_TOKEN` and `HF_TOKEN`,
add your Git and SSH configuration.

*Also, ensure that you have the necessary SSH keys set up, including `~/.ssh/prime` for connecting to the Prime instance and `~/.ssh/github-personal` for GitHub access.*

**3. Run the Setup Script**

Execute the `setup.sh` script to set up the environment directly from your machine. Simply pass in the SSH string for your instance (to see how to obtain this, refer to the [docs](https://docs.primeintellect.ai/quickstart)):

```bash
bash scripts/setup.sh "<SSH_STRING>"
```

The script will transfer the necessary files to the Prime instance and execute a setup script on the remote server. It will clone the repository using git, install Miniconda, create and activate a conda environment, install required dependencies, log in to Weights & Biases and Hugging Face.

**4. Connecting to Server**

After the setup script completes, you can connect to the prepared server using:

```bash
ssh -p <port> <user>@<host>
```

Once connected, navigate to the project directory and activate the virtual environment:

```bash
cd swarm && conda activate swarm
```

You are now ready to start training models! 🚀

**5. Cleaning Up (Optional)**

If you need to remove the setup, you can use the cleanup scripts from your local machine or the remote server. From your local machine, run:

```bash
bash scripts/cleanup.sh "<SSH_STRING>"
```

Or, from within the Prime instance, run:

```bash
bash ~/cleanup_remote.sh
```

These will remove the Miniconda installation, the project directory, and other setup files from the server.