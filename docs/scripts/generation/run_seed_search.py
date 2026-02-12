import os
import yaml
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
import sys

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_experiment(seed, k, base_config_path, base_config):
    """
    Sets up and runs a single experiment for a given seed and k.
    """
    # Create specific run name
    run_name = f"Exp_TABS_k{k}_s{seed}"
    
    # Create a temporary config file for this run
    temp_config_dir = "temp_configs"
    os.makedirs(temp_config_dir, exist_ok=True)
    temp_config_path = os.path.join(temp_config_dir, f"{run_name}.yaml")
    
    # Modify config
    current_config = base_config.copy()
    current_config['run_name'] = run_name
    current_config['data']['seed'] = seed
    current_config['model']['params']['num_selected_bands'] = k
    
    # Ensure model name is TABS (just to be safe/consistent, though ASRAN_LBS works now)
    current_config['model']['name'] = 'TABS' 

    save_config(current_config, temp_config_path)
    
    print(f"Starting experiment: {run_name} (Seed: {seed}, k: {k})")
    
    # Execute the training script
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    
    log_file = os.path.join(temp_config_dir, f"{run_name}.log")
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(
                [sys.executable, "scripts/run.py", "--config", temp_config_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                env=env
            )
        print(f"Finished experiment: {run_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error in experiment {run_name}: {e}")
    finally:
        # Cleanup temp config if desired, or keep for debugging
        # os.remove(temp_config_path)
        pass

def main():
    base_config_path = "training_runs/0.7600_0.2_Exp_TABS_k7_s42_d4i9_20251225_110411/config.yaml"
    
    if not os.path.exists(base_config_path):
        print(f"Error: Base config file not found at {base_config_path}")
        return

    base_config = load_config(base_config_path)
    
    seeds = range(44, 101) # 44 to 100
    k_values = [3, 5, 7]
    
    tasks = []
    for seed in seeds:
        for k in k_values:
            tasks.append((seed, k))
            
    # Max parallel processes
    max_workers = 21
    
    print(f"Total experiments to run: {len(tasks)}")
    print(f"Running with {max_workers} parallel workers.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for seed, k in tasks:
            futures.append(executor.submit(run_experiment, seed, k, base_config_path, base_config))
            
        # Wait for all to complete
        for future in futures:
            future.result()

    print("All experiments completed.")

if __name__ == "__main__":
    main()
