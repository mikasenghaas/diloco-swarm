import re
import shutil
import subprocess

import torch
import pytest
    
TIMEOUT = 30
LOG_DIR = "/tmp/swarm/test_memorize"
CMD = lambda num_processes, num_stages, device, run_id: [
    "torchrun",
    "--nproc_per_node", str(num_processes),
    "src/train.py",
    "--swarm.num_stages", str(num_stages),
    "--device", str(device),
    "--model", "@configs/model/gpt2-tiny.toml",
    "--data", "@configs/data/memorize.toml",
    "--data.tokenize", "true",
    "--data.num_workers", "1",
    "--data.pin_memory", "false",
    "--data.seq_length", "128",
    "--train.inner_optimizer", "@configs/optimizer/adamw.toml",
    "--train.outer_optimizer", "@configs/optimizer/none.toml",
    "--train.inner_optimizer.lr", "0.006",
    "--train.max_steps", "40",
    "--train.batch_size", "1",
    "--train.micro_batch_size", "1",
    "--logging.log_dir", LOG_DIR,
    "--logging.run_id", run_id,
    "--amp.enable", "false",
    "--eval.enable", "false",
    "--logging.wandb.enable", "false",
]

@pytest.fixture(params=["cpu", "cuda"], scope="session")
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return request.param

@pytest.fixture(params=[
    {"name": "single_process", "num_processes": 1, "num_stages": 1},
    {"name": "data_parallel", "num_processes": 2, "num_stages": 1},
    {"name": "pipeline_parallel", "num_processes": 2, "num_stages": 2},
    {"name": "large_pipeline_parallel", "num_processes": 3, "num_stages": 3},
    {"name": "swarm", "num_processes": 4, "num_stages": 2},
    {"name": "large_swarm", "num_processes": 9, "num_stages": 3},
], ids=lambda x: x["name"], scope="session")
def config(request):
    return request.param

@pytest.fixture(scope="session")
def process(device, config):
    """Start memorize training process."""
    # Create run id
    run_id = str(device) + "-" + config["name"]

    # Remove previous log directory if it exists
    shutil.rmtree(f"{LOG_DIR}/{run_id}", ignore_errors=True)

    command = CMD(num_processes=config["num_processes"], num_stages=config["num_stages"], device=device, run_id=run_id)
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

def test_memorization(process):
    sentence = "I am a large language model and I can memorize this sentence."
    log_file = f"{LOG_DIR}/{process[1]}/master.log"
    with open(log_file, "r") as f:
        logs = f.read()
    assert len(re.findall(sentence, logs)) > 0, "Memorized sentence not found in logs"
