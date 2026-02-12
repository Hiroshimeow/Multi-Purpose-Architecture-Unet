import os
import yaml
import subprocess
import time
import sys
from datetime import datetime
import copy

# --- CONFIGURATION ---
MAX_CONCURRENT_WORKERS = 30
PYTHON_EXEC = os.path.abspath("./venv/bin/python")
RUN_SCRIPT = "scripts/run.py"
EPOCHS = 400
POLL_INTERVAL_SECONDS = 30  # Check for completed jobs every 30 seconds

# --- EXPERIMENT MATRIX ---
SEEDS = [42]
K_VALUES = [3, 5, 7]

METHODS = [
    # {"name": "TABS",        "model": "TABS",       "type": "end-to-end"},
    {"name": "SVM-RFE", "model": "StandardUNet", "type": "baseline", "ranking": "SVM"},
    {
        "name": "Entropy",
        "model": "StandardUNet",
        "type": "baseline",
        "ranking": "Entropy",
    },
    {
        "name": "Correlation",
        "model": "StandardUNet",
        "type": "baseline",
        "ranking": "Correlation",
    },
    {
        "name": "Fisher",
        "model": "StandardUNet",
        "type": "baseline",
        "ranking": "Fisher",
    },
    {"name": "SAD", "model": "StandardUNet", "type": "baseline", "ranking": "SAD"},
]

ARCHITECTURES = {
    "standard": {"lightweight": False},
    "lightweight_ds": {"lightweight": True},
}

# --- BASE CONFIG (in-script to avoid file dependencies) ---
BASE_CONFIG = {
    "base_output_dir": "./training_runs",
    "data": {
        "dir": "Image_dataset",
        "file_extension": "mat",
        "normalization": "min-max",
        "class_mapping": {
            1: 0,
            2: 1,
            3: 2,
            5: 3,
            0: 4,
            4: 4,
            6: 4,
            7: 4,
            8: 4,
            9: 4,
            10: 4,
        },
        "patching": {"use_patching": True, "patch_size": 224, "stride": 192},
    },
    "augmentation": {"use_augmentation": True},
    "model": {
        "class_names": ["Road", "RoadLine", "Vegetation", "Sky", "NoDrivable"],
        "params": {
            "num_classes": 5,
            "initial_filters": 6,  # Corresponds to ~0.17M model
            "depth": 4,
            "bilinear": True,
        },
    },
    "loss": {"name": "CombinedLoss", "params": {"alpha": 0.4, "beta": 0.6}},
    "training": {"batch_size": 16, "use_amp": True, "early_stopping_patience": 80},
    "optimizer": {"name": "AdamW", "params": {"lr": 0.001, "weight_decay": 0.0001}},
    "scheduler": {"name": "CosineAnnealingLR", "params": {"eta_min": 1e-06}},
}


# --- DIRECTORIES ---
CONFIG_DIR = "temp_master_configs"
LOG_DIR = "master_logs"
COMPLETED_LOG = "master_completed_jobs.log"


def main():
    # --- SETUP ---
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- JOB QUEUE GENERATION ---
    jobs_queue = []
    for k in K_VALUES:
        for seed in SEEDS:
            for method in METHODS:
                for arch_name, arch_params in ARCHITECTURES.items():
                    job = {
                        "method": method,
                        "k": k,
                        "seed": seed,
                        "arch_name": arch_name,
                        "arch_params": arch_params,
                    }
                    jobs_queue.append(job)

    total_jobs = len(jobs_queue)
    print(f"Generated {total_jobs} total jobs.")

    # --- PROCESS MANAGEMENT ---
    running_processes = []

    # Prepare environment for subprocesses
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    cudnn_path = os.path.abspath("venv/lib/python3.10/site-packages/nvidia/cudnn/lib")
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cudnn_path}:{current_ld}" if current_ld else cudnn_path

    # --- MAIN LOOP ---
    completed_count = 0
    while jobs_queue or running_processes:
        # Check for completed processes
        finished_indices = []
        for i, p_info in enumerate(running_processes):
            if p_info["process"].poll() is not None:
                finished_indices.append(i)
                completed_count += 1
                with open(COMPLETED_LOG, "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file = p_info["log_file"]
                    return_code = p_info["process"].returncode
                    status = "SUCCESS" if return_code == 0 else f"FAIL ({return_code})"
                    f.write(f"{timestamp} | {p_info['name']} | {status}\n")
                print(
                    f"({completed_count}/{total_jobs}) COMPLETED: {p_info['name']} with code {p_info['process'].returncode}"
                )
                # Clean up the temp config file for the completed job
                if os.path.exists(p_info["config_path"]):
                    os.remove(p_info["config_path"])

        # Remove finished processes from the list
        for i in sorted(finished_indices, reverse=True):
            del running_processes[i]

        # Launch new processes
        while len(running_processes) < MAX_CONCURRENT_WORKERS and jobs_queue:
            job = jobs_queue.pop(0)
            method = job["method"]
            k = job["k"]
            seed = job["seed"]
            arch_name = job["arch_name"]
            arch_params = job["arch_params"]

            run_name = f"Exp_{method['name']}_k{k}_s{seed}_{arch_name}"

            config = copy.deepcopy(BASE_CONFIG)
            config["run_name"] = run_name
            config["data"]["seed"] = seed
            config["training"]["num_epochs"] = EPOCHS

            # Set model and architecture parameters
            config["model"]["name"] = method["model"]
            config["model"]["params"]["in_channels"] = k
            config["model"]["params"]["lightweight"] = arch_params["lightweight"]
            config["model"]["params"]["use_sa"] = True  # Enable Spatial Attention

            # Set ranking method for baselines, not needed for TABS
            if method["type"] == "baseline":
                config["model"]["ranking_method"] = method["ranking"]

            if method["type"] == "end-to-end":
                # TABS model has its own band selector and takes all bands as input
                config["model"]["params"]["in_channels"] = 25
                config["model"]["params"]["num_select_bands"] = k

            temp_config_path = os.path.join(CONFIG_DIR, f"{run_name}.yaml")
            with open(temp_config_path, "w") as f:
                yaml.dump(config, f)

            log_file_path = os.path.join(LOG_DIR, f"{run_name}.log")
            cmd = [PYTHON_EXEC, "-u", RUN_SCRIPT, "--config", temp_config_path]

            print(
                f"LAUNCHING ({len(running_processes) + 1}/{MAX_CONCURRENT_WORKERS}): {run_name} -> {log_file_path}"
            )
            with open(log_file_path, "w") as out_log:
                p = subprocess.Popen(
                    cmd, stdout=out_log, stderr=subprocess.STDOUT, env=env
                )
                running_processes.append(
                    {
                        "process": p,
                        "name": run_name,
                        "log_file": log_file_path,
                        "config_path": temp_config_path,
                    }
                )

        if not jobs_queue and not running_processes:
            break

        print(
            f"--- {len(running_processes)} jobs running. {len(jobs_queue)} jobs in queue. Polling in {POLL_INTERVAL_SECONDS}s. ---"
        )
        time.sleep(POLL_INTERVAL_SECONDS)

    print("\n--- All jobs completed. ---")


if __name__ == "__main__":
    main()
