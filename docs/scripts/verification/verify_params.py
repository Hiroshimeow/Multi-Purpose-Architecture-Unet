import os
import yaml
import subprocess
import sys

# --- CONFIG ---
PYTHON_EXEC = os.path.abspath("./venv/bin/python") 
RUN_SCRIPT = "run.py"
BASE_CONFIG_PATH = "training_runs/0.7632 ASRAN_LBS_p224_e192_small_20251014_203805/config.yaml"
TARGET_FILTERS = 7 # Test with 7

# --- SCRIPT ---
print(f"--- Verifying parameter count with base_filters = {TARGET_FILTERS} ---")

# Load and modify config
try:
    config = yaml.safe_load(open(BASE_CONFIG_PATH, "r"))
except FileNotFoundError:
    print(f"Error: Base config not found at {BASE_CONFIG_PATH}")
    sys.exit(1)

config["model"]["params"]["base_filters"] = 7
config["model"]["params"]["use_sa"] = True # Test with Spatial Attention enabled
config["run_name"] = "Param_Count_Verification_Run_SA"
config["training"]["num_epochs"] = 1 # Only need to run long enough to init model

# Use TABS for the test, k=7
config["model"]["name"] = "ASRAN_LBS"
config["model"]["params"]["in_channels"] = 25
config["model"]["params"]["num_select_bands"] = 7
if "base_filters" in config["model"]["params"]:
    del config["model"]["params"]["base_filters"] # This is no longer used by the hardcoded UnetBase
if "selected_bands_indices" in config["model"]["params"]:
    del config["model"]["params"]["selected_bands_indices"]

# Save temp config
temp_config_path = "temp_config_param_check.yaml"
with open(temp_config_path, "w") as f:
    yaml.dump(config, f)

# Prepare command
cmd = [PYTHON_EXEC, "-u", RUN_SCRIPT, "--config", temp_config_path]

# Prepare environment
env = os.environ.copy()
cudnn_path = os.path.abspath("venv/lib/python3.10/site-packages/nvidia/cudnn/lib")
current_ld = env.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = f"{cudnn_path}:{current_ld}"

# Run and capture output
print("Starting verification run...")
try:
    result = subprocess.run(cmd, check=False, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    output = result.stdout + result.stderr
    for line in output.splitlines():
        if "Model Stats:" in line:
            print(line)
            # Try to find a line with Params (M) and extract the value
            parts = line.split(',')
            for part in parts:
                if 'M Params' in part:
                    param_count = float(part.strip().split(' ')[0])
                    if 0.25 <= param_count <= 0.29:
                        print(f"SUCCESS: Found parameter count {param_count}M, which is close to the 0.27M target.")
                    else:
                        print(f"INFO: Parameter count is {param_count}M. This is not the target 0.27M.")
            break
    else:
        print("Could not find 'Model Stats' line. Full output:")
        print(output)

except Exception as e:
    print(f"Execution Error: {e}")
finally:
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)