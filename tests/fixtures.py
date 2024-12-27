import pytest
import torch

DEVICES = ["cpu", "cuda"]
CONFIGS = [
    {"name": "single_process", "num_processes": 1, "num_stages": 1},
    {"name": "data_parallel", "num_processes": 2, "num_stages": 1},
    {"name": "large_data_parallel", "num_processes": 3, "num_stages": 1},
    {"name": "pipeline_parallel", "num_processes": 2, "num_stages": 2},
    {"name": "large_pipeline_parallel", "num_processes": 3, "num_stages": 3},
    {"name": "swarm", "num_processes": 4, "num_stages": 2},
    {"name": "large_swarm", "num_processes": 9, "num_stages": 3},
]

@pytest.fixture(params=DEVICES, scope="session")
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return request.param

@pytest.fixture(params=CONFIGS, scope="session", ids=lambda config: config["name"])
def config(request):
    return request.param
