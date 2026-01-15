
import os
import yaml
import sys

# --- CONFIG FOR A SINGLE DEBUG RUN ---
PYTHON_EXEC = os.path.abspath("./venv/bin/python")
RUN_SCRIPT = "run.py"
BASE_CONFIG_PATH = "config_base_170k.yaml"
EPOCHS = 400
SEED = 42

# --- Single Job to Debug ---
k = 3
method = {"name": "TABS", "model": "TABS", "type": "end-to-end"}
config_set = {"name": "d4i9", "params": {"depth": 4, "initial_filters": 9}}

def main():
    """
    Generates the config and command for a single debug job.
    """
    print("--- Preparing a single job for debugging ---")
    
    # Load Band Rankings
    try:
        RANKINGS = {}
        with open("band_rankings.txt", "r") as f:
            for line in f:
                if ":" in line:
                    name, indices = line.strip().split(":")
                    RANKINGS[name] = [int(x) for x in indices.split(",")]
    except FileNotFoundError:
        print("FATAL: band_rankings.txt not found.")
        sys.exit(1)

    # Load base config template
    base_config = yaml.safe_load(open(BASE_CONFIG_PATH, "r"))
    
    set_name = config_set["name"]
    model_base_params = config_set["params"]
    output_dir = os.path.join("run_test2", set_name)
    os.makedirs(output_dir, exist_ok=True)

    method_name = method["name"]
    run_name = f"Exp_{method_name}_k{k}_s{SEED}_{set_name}"
    
    # --- Create Job-Specific Config ---
    config = base_config.copy()
    config["run_name"] = run_name
    config["data"]["seed"] = SEED
    config["training"]["num_epochs"] = EPOCHS

    model_params = model_base_params.copy()
    model_params.update({"num_classes": 5, "bilinear": True})

    if method["type"] == "end-to-end":
        config["model"]["name"] = "TABS"
        model_params["in_channels"] = 25
        model_params["num_selected_bands"] = k
    else: # Should not happen for this debug run
        pass
    
    config["model"]["params"] = model_params

    # Write temp config
    temp_config_path = os.path.join(output_dir, f"{run_name}.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # --- Generate Command ---
    cmd = f"{PYTHON_EXEC} -u {RUN_SCRIPT} --config {temp_config_path}"
    
    print(f"\nGenerated Config: {temp_config_path}")
    print(f"\nRUN THIS COMMAND TO DEBUG:\n\n{cmd}\n")

if __name__ == "__main__":
    main()
