import subprocess
import sys
import pytest

def test_huggingface():
    """Test if Hugging Face is logged in."""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], capture_output=True, text=True)
        assert "Not logged in" not in result.stdout, "Not logged in to Hugging Face"
    except FileNotFoundError:
        pytest.fail("Hugging Face CLI is not installed.")

def test_wandb():
    """Test if Weights & Biases is logged in."""
    try:
        result = subprocess.run(['wandb', 'login'], stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
        assert "Currently logged in" in result.stdout, "Not logged in to Weights & Biases"
    except FileNotFoundError:
        pytest.fail("Weights & Biases CLI is not installed.")

def test_dependencies():
    """Test if all dependencies in requirements.txt are installed."""
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()

    for package in requirements:
        if package.startswith('git') or "[" in package: continue
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'show', package])
        except subprocess.CalledProcessError:
            pytest.fail(f"Package {package} is not installed.")
