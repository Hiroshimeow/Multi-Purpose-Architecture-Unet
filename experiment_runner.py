import os
import yaml
import subprocess
import time

# Configuration
PYTHON_EXEC = "python3" 
RUN_SCRIPT = "run.py"
BASE_CONFIG_PATH = "training_runs/0.7632 ASRAN_LBS_p224_e192_small_20251014_203805/config.yaml"
EPOCHS = 400  # Main experiment settings

# Experiment Matrix
SEEDS = [42, 100, 2024]
K_VALUES = [3, 5, 7]

# Load Rankings
RANKINGS = {}
if os.path.exists("band_rankings.txt"):
    with open("band_rankings.txt", "r") as f:
        for line in f:
            if ":" in line:
                name, indices = line.strip().split(":")
                RANKINGS[name] = [int(x) for x in indices.split(",")]
else:
    print("Warning: band_rankings.txt not found. Using dummy rankings.")
    RANKINGS["SVM"] = list(range(25))
    RANKINGS["Entropy"] = list(range(25))

# Methods to run
METHODS = [
    {"name": "TABS", "model": "ASRAN_LBS", "type": "end-to-end"},
    {"name": "SVM-RFE", "model": "StandardUNet", "type": "baseline", "ranking": "SVM"},
    {"name": "Entropy", "model": "StandardUNet", "type": "baseline", "ranking": "Entropy"}
]

def load_base_config():
    with open(BASE_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def run_experiment():
    for method in METHODS:
        for k in K_VALUES:
            # For baselines, run only 1 seed to save time in this demo, unless TABS
            current_seeds = SEEDS if method["name"] == "TABS" else [SEEDS[0]]
            
            for seed in current_seeds:
                print(f"\n>>> Starting Experiment: Method={method['name']}, k={k}, Seed={seed}")
                
                # Prepare Config
                try:
                    config = load_base_config()
                except FileNotFoundError:
                    print("Base config not found!")
                    return

                config["training"]["num_epochs"] = EPOCHS
                config["data"]["seed"] = seed
                config["run_name"] = f"Exp_{method['name']}_k{k}_s{seed}"
                
                # Model Specific Setup
                if method["type"] == "end-to-end":
                    config["model"]["name"] = "ASRAN_LBS"
                    config["model"]["params"]["in_channels"] = 25
                    config["model"]["params"]["num_select_bands"] = k
                    if "selected_bands_indices" in config["model"]["params"]:
                        del config["model"]["params"]["selected_bands_indices"]
                        
                elif method["type"] == "baseline":
                    config["model"]["name"] = "StandardUNet"
                    config["model"]["params"]["in_channels"] = k 
                    
                    if method["ranking"] in RANKINGS:
                        top_k_indices = RANKINGS[method["ranking"]][:k]
                        top_k_indices = sorted(top_k_indices)
                        config["model"]["params"]["selected_bands_indices"] = top_k_indices
                    else:
                        print(f"Ranking {method['ranking']} not found!")
                        continue
                    
                # Save Temp Config
                temp_config_path = f"temp_config_{method['name']}_{k}_{seed}.yaml"
                with open(temp_config_path, "w") as f:
                    yaml.dump(config, f)
                
                # Execute Run
                cmd = [
                    PYTHON_EXEC, RUN_SCRIPT,
                    "--config", temp_config_path,
                    "--epochs", str(EPOCHS)
                ]
                
                try:
                    # Force CPU
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = ""
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    
                    if result.returncode == 0:
                        print(f"✅ Experiment Successful.")
                    else:
                        print(f"❌ Experiment Failed.")
                        print("--- STDOUT ---")
                        print(result.stdout[-2000:])
                        print("--- STDERR ---")
                        print(result.stderr[-1000:]) 
                        
                except Exception as e:
                    print(f"❌ Execution Error: {e}")
                finally:
                    if os.path.exists(temp_config_path):
                        os.remove(temp_config_path)

if __name__ == "__main__":
    run_experiment()