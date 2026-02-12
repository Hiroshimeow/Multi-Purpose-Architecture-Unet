# Configuration
PYTHON_EXEC = "./venv/bin/python" 
RUN_SCRIPT = "run.py"
BASE_CONFIG_PATH = "config_base_170k.yaml" # Use the 0.17M parameter config
EPOCHS = 400

# Experiment Matrix
SEEDS = [42, 84, 2025]
K_VALUES = [3, 5, 7] 

# ... (rest of the script is the same as the parallel runner)
