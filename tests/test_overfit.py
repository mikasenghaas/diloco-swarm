import re
import shutil
import subprocess

import pytest
    
TIMEOUT = 30
LOG_DIR = "/tmp/swarm/test"
CMD = lambda num_processes, num_stages, device, run_id: [
    "torchrun",
    "--nproc_per_node", str(num_processes),
    "src/train.py",
    "--swarm.num_stages", str(num_stages),
    "--device", str(device),
    "--model", "@configs/model/gpt2-tiny.toml",
    "--data", "@configs/data/memorize.toml",
    "--data.seq_length", "128",
    "--train.max_epochs", "40",
    "--train.batch_size", "1",
    "--train.micro_batch_size", "1",
    "--train.optimizer.lr", "0.006",
    "--amp.enable", "false",
    "--logging.log_dir", LOG_DIR,
    "--logging.run_id", run_id,
    "--sample.enable", "true",
    "--eval.enable", "true",
    "--logging.wandb.enable", "false",
]

@pytest.fixture(params=["cpu", "cuda"], scope="session")
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return request.param

@pytest.fixture(scope="session")
@pytest.mark.parametrize("config", [
    {"name": "single_process", "num_processes": 1, "num_stages": 1},
    {"name": "data_parallel", "num_processes": 2, "num_stages": 1},
    {"name": "pipeline_parallel", "num_processes": 2, "num_stages": 2},
    {"name": "large_pipeline_parallel", "num_processes": 3, "num_stages": 3},
    {"name": "swarm", "num_processes": 4, "num_stages": 2},
    {"name": "large_swarm", "num_processes": 9, "num_stages": 3},
])
def overfit_process(device, config):
    """Start overfit training process."""
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

def test_no_errors(overfit_process):
    process = overfit_process[0]
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"

def test_memorization(overfit_process):
    sentence = "I am a large language model and I can memorize this sentence."
    log_file = f"{LOG_DIR}/{overfit_process[1]}/master.log"
    with open(log_file, "r") as f:
        logs = f.read()
    assert len(re.findall(sentence, logs)) == 5, "Memorized sentence not found in logs"