import os
import yaml
import subprocess
import time
import sys

# Configuration
PYTHON_EXEC = os.path.abspath("./venv/bin/python") 
RUN_SCRIPT = "run.py"
BASE_CONFIG_PATH = "training_runs/0.7632 ASRAN_LBS_p224_e192_small_20251014_203805/config.yaml"
EPOCHS = 2

# Load Rankings (make sure band_rankings.txt exists from previous run)
RANKINGS = {}
if os.path.exists("band_rankings.txt"):
    with open("band_rankings.txt", "r") as f:
        for line in f:
            if ":" in line:
                name, indices = line.strip().split(":")
                RANKINGS[name] = [int(x) for x in indices.split(",")]
else:
    print("Error: band_rankings.txt not found. Exiting.")
    sys.exit(1)

method = {"name": "Entropy", "model": "StandardUNet", "type": "baseline", "ranking": "Entropy"}
k = 3
seed = 42

print(f"\n>>> Starting Debug Experiment: Method={method['name']}, k={k}, Seed={seed}")

config = yaml.safe_load(open(BASE_CONFIG_PATH, "r"))
config["training"]["num_epochs"] = EPOCHS
config["data"]["seed"] = seed
config["run_name"] = f"Debug_{method['name']}_k{k}_s{seed}"

config["model"]["name"] = "StandardUNet"
config["model"]["params"]["in_channels"] = k 

top_k_indices = RANKINGS[method["ranking"]][:k]
top_k_indices = sorted(top_k_indices)
config["model"]["params"]["selected_bands_indices"] = top_k_indices

temp_config_path = f"temp_config_debug.yaml"
with open(temp_config_path, "w") as f:
    yaml.dump(config, f)

cmd = [
    PYTHON_EXEC, "-u", RUN_SCRIPT,
    "--config", temp_config_path,
    "--epochs", str(EPOCHS)
]

env = os.environ.copy()
# Ensure venv's site-packages are in PYTHONPATH
venv_python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
env["PYTHONPATH"] = f"{os.path.abspath('venv/lib/')}/{venv_python_version}/site-packages:{env.get('PYTHONPATH', '')}"

cudnn_path = os.path.abspath("venv/lib/python3.10/site-packages/nvidia/cudnn/lib")
current_ld = env.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = f"{cudnn_path}:{current_ld}"

try:
    result = subprocess.run(cmd, check=False, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    if result.returncode == 0:
        print(f"✅ Debug Experiment Successful.")
    else:
        print(f"❌ Debug Experiment Failed with return code {result.returncode}.")
except Exception as e:
    print(f"❌ Execution Error: {e}")
finally:
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)