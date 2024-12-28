import shutil
import subprocess

import pytest
    
TIMEOUT = 120
LOG_DIR = "/tmp/swarm/test_train"
CMD = lambda num_processes, num_stages, run_id: [
    "torchrun",
    "--nproc_per_node", str(num_processes),
    "src/train.py",
    "--swarm.num_stages", str(num_stages),
    "--device", "cuda",
    "--model", "@configs/model/gpt2-small.toml",
    "--data", "@configs/data/wikitext.toml",
    "--train.inner_optimizer", "@configs/optimizer/adamw.toml",
    "--train.outer_optimizer", "@configs/optimizer/none.toml",
    "--train.inner_optimizer.lr", "6e-4",
    "--data.seq_length", "1024",
    "--train.batch_size", "512",
    "--train.micro_batch_size", "4",
    "--train.max_steps", "4",
    "--logging.log_dir", LOG_DIR,
    "--logging.run_id", run_id,
    "--sample.enable", "true",
    "--sample.every_n_steps", "2",
    "--eval.enable", "true",
    "--eval.max_steps", "10",
    "--eval.every_n_steps", "2",
    "--logging.wandb.enable", "false",
]

@pytest.fixture(params=[
    {"name": "single_process", "num_processes": 1, "num_stages": 1},
    {"name": "data_parallel", "num_processes": 2, "num_stages": 1},
    {"name": "pipeline_parallel", "num_processes": 2, "num_stages": 2},
    {"name": "swarm", "num_processes": 4, "num_stages": 2},
], ids=lambda x: x["name"], scope="session")
def config(request):
    return request.param

@pytest.fixture(scope="session")
def process(config):
    """Start training process."""
    # Create run id
    run_id = config["name"]

    # Remove previous log directory if it exists
    shutil.rmtree(f"{LOG_DIR}/{run_id}", ignore_errors=True)

    command = CMD(num_processes=config["num_processes"], num_stages=config["num_stages"], run_id=run_id)
    process = subprocess.Popen(command)
    
    try:
        process.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait()
        raise TimeoutError(f"Process did not complete within {TIMEOUT} seconds")

    yield process, run_id
    
    process.terminate()
    process.wait()

def test_no_errors(process):
    assert process[0].returncode == 0, f"Process failed with return code {process[0].returncode}"