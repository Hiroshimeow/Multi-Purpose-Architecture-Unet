import os
import yaml
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_experiment(seed, k, base_config):
    """
    Sets up and runs a single experiment for a given seed and k.
    """
    # Create specific run name
    run_name = f"Exp_TABS_k{k}_s{seed}"
    
    # Create a temporary config file for this run
    temp_config_dir = "temp_configs_clean"
    os.makedirs(temp_config_dir, exist_ok=True)
    temp_config_path = os.path.join(temp_config_dir, f"{run_name}.yaml")
    
    # Modify config
    current_config = base_config.copy()
    current_config['run_name'] = run_name
    current_config['data']['seed'] = seed
    current_config['model']['params']['num_selected_bands'] = k
    
    # Ensure model name is TABS
    current_config['model']['name'] = 'TABS' 

    save_config(current_config, temp_config_path)
    
    print(f"Starting experiment: {run_name} (Seed: {seed}, k: {k})")
    
    # Execute the training script
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + (":" + env.get("PYTHONPATH", ""))
    
    log_file = os.path.join(temp_config_dir, f"{run_name}.log")
    
    try:
        with open(log_file, 'w') as f:
            # Run with unbuffered output
            subprocess.run(
                [sys.executable, "-u", "scripts/run.py", "--config", temp_config_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                env=env
            )
        print(f"Finished experiment: {run_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error in experiment {run_name}: {e}")
        # Print tail of log
        try:
            with open(log_file, 'r') as f:
                print(f"Tail of log {run_name}:")
                print("".join(f.readlines()[-10:]))
        except:
            pass

def main():
    base_config_path = "training_runs/Exp_TABS_k7_s54_20260102_132253/config.yaml"
    
    if not os.path.exists(base_config_path):
        print(f"Error: Base config file not found at {base_config_path}")
        return

    base_config = load_config(base_config_path)
    
    # New clean seeds
    seeds = [101, 102, 103, 104, 105]
    k_values = [3, 5, 7]
    
    tasks = []
    for seed in seeds:
        for k in k_values:
            tasks.append((seed, k))
            
    # Sequential execution for safety (max_workers=1)
    max_workers = 1
    
    print(f"Total experiments to run: {len(tasks)}")
    print(f"Running with {max_workers} parallel workers.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for seed, k in tasks:
            futures.append(executor.submit(run_experiment, seed, k, base_config))
            
        for future in futures:
            future.result()

    print("All experiments completed.")

if __name__ == "__main__":
    main()
