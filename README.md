# DiLoCo-SWARM

This repository contains the full experiment code and results for validating DiLoCo-SWARM, a distributed training method that combines the pipeline and data parallelism of [SWARM](https://arxiv.org/abs/2301.11913) with [DiLoCo](https://arxiv.org/abs/2311.08105)'s reduced-frequency gradient synchronization.

## 🔗 Shortcuts

- 📄 Read the [report](report.pdf)
- 📊 View raw and analyzed experiment results in [W&B](https://wandb.ai/mikasenghaas/diloco-swarm) and this  [notebook](notebooks/3.0-results.ipynb)
- 🧠 Understand DP, PP, DiLoCo, SWARM, and their interplay in this pure PyTorch
single-file [training script]() - distilled from this codebase
- 💻 Replicate the experiments following the [setup](#setup) and [replicating experiments](#replicating-experiments) sections below.


## ⚙️ Setup

This project should run on any machine with GPU support, and PyTorch installed.
To be safe, follow my setup, which is described below. Get yourself access to
some GPUs via [Prime Compute](https://www.app.primeintellect.com/).

**1. Clone the Repository**

```bash
gh repo clone mikasenghaas/diloco-swarm && cd diloco-swarm
```

**2. Configure Environment**

Configure environment variables for W&B, HF, Git, and SSH. To make this simpler, copy the example files I prepared, and fill in your info.

```bash
cp .env.example .env
cp .gitconfig.example .gitconfig
cp .sshconfig.example .sshconfig
```

**3. Run the Setup Script**

Execute the `setup.sh` script to set up the environment directly from your machine. You will need to know the user, host name, path to the persistent directory, and optionally a port number. All of these will be provided by Prime Compute once you have a running instance. If you have problems, check the [docs](https://docs.primeintellect.ai/quickstart) for more information.

```bash
bash scripts/setup.sh <USER> <HOST> <PERSISTENT_DIR> [<PORT>]
```

The script will transfer the necessary files to the remote server and execute a setup script. It will clone the repository, create and activate a virtual environment, install dependencies, and finally log in to W&B and HF. For details, check the [setup script](scripts/setup.sh) and [remote setup script](scripts/setup_remote.sh).

**4. Connecting to Server**

After the setup script completes, you can connect to the prepared server using:

```bash
ssh -p <port> <user>@<host>
```

Once connected, navigate to the project directory and activate the virtual environment:

```bash
cd diloco-swarm && conda activate diloco-swarm
```

You are now ready to start training models! 🚀

**5. Verify Setup**

To verify that the setup was successful, you can try running any of the experiments in the [experiments](experiments) directory, as outlined in this this [section](#replicating-experiments). Otherwise, you can run part of the test suite. This requires `pytest` to be installed.

```bash
cd diloco-swarm && pip install -r requirements.dev.txt
```

Once completed, you can run a simple test that checks whether the setup was successful.

```bash
pytest tests/test_setup.py
```


**6. Cleaning Up (Optional)**

If you need to remove the setup, you can use the cleanup scripts from your local machine or the remote server. From your local machine, run:

```bash
bash scripts/cleanup.sh "<SSH_STRING>"
```

Or, from within the Prime instance, run:

```bash
bash ~/cleanup_remote.sh
```

These will remove the Miniconda installation, the project directory, and other setup files from the server.

## 🚀 Replicating Experiments

All experiments are bundled in simple bash scripts in the
[experiments](experiments) directory:

1. [Experiment 1](): Main experiment testing DiLoCo-SWARM against Single-GPU and SWARM baselines.
2. [Experiment 2](): Communication frequency ablation testing the impact of reduced-frequency gradient synchronization.
3. [Experiment 3](): Model size ablation testing the impact of model size 

To run any experiment, simply execute the corresponding script. For example, to
replicate the main experiment, execute:

```bash
bash experiments/experiment1.sh
```

*NB: Some of these experiments run for multiple hours, or even days. Make sure to check the output of the experiment script to ensure it is running as expected.*