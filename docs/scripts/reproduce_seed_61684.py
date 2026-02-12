import os
import yaml
import subprocess
import time
import sys
import glob

# --- CONFIGURATION ---
TARGET_SEED = 61684
EPOCHS = 400
MAX_PARALLEL_JOBS = 24 # Full power execution
# Rankings from calculate_bands.py (Best to Worst)
RANKINGS = {
    'Entropy': [7, 11, 6, 2, 8, 22, 10, 5, 23, 21, 3, 12, 20, 1, 17, 16, 0, 13, 18, 15, 4, 19, 24, 9, 14],
    'SVM-RFE': [17, 13, 15, 4, 18, 9, 14, 8, 1, 22, 20, 7, 0, 23, 10, 3, 5, 6, 11, 19, 24, 16, 21, 12, 2],
    'Fisher':  [10, 5, 20, 6, 11, 9, 8, 21, 19, 22, 12, 7, 23, 15, 17, 16, 18, 3, 14, 13, 2, 1, 0, 4, 24],
    'Correlation': [6, 7, 11, 8, 5, 17, 21, 3, 18, 2, 1, 15, 10, 0, 22, 23, 16, 20, 12, 19, 24, 9, 4, 14, 13],
    'SAD':     [7, 6, 11, 5, 8, 2, 22, 21, 3, 23, 10, 20, 1, 0, 12, 13, 16, 17, 18, 4, 15, 9, 24, 19, 14]
}

# Generate Experiment Matrix: 6 Methods x 3 Ks = 18 Jobs
METHODS = ['TABS', 'SVM-RFE', 'Entropy', 'Fisher', 'Correlation', 'SAD']
K_VALUES = [3, 5, 7]

JOBS = []
for m in METHODS:
    for k in K_VALUES:
        JOBS.append({'method': m, 'k': k, 'seed': TARGET_SEED})

def create_config(job):
    method = job['method']
    k = job['k']
    seed = job['seed']
    
    # Short name for file paths
    run_name = f"S{seed}_{method}_k{k}"
    if EPOCHS < 10: run_name += "_TEST"
    
    config = {
        'run_name': run_name,
        'base_output_dir': './training_runs',
        'class_names': ["Road", "RoadLine", "Vegetation", "Sky", "NoDrivable"],
        'data': {
            'dir': "Image_dataset",
            'normalization': 'min-max',
            'class_mapping': {1: 0, 2: 1, 3: 2, 5: 3, 0: 4, 4: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4},
            'patching': {'patch_size': 224, 'stride': 192},
            'seed': seed,
            'val_split': 0.2,
            'test_split': 0.2
        },
        'augmentation': {
            'shift_scale_rotate': {'p': 0.5, 'shift_limit': 0.0625, 'scale_limit': 0.1, 'rotate_limit': 15},
            'random_brightness_contrast': {'p': 0.5}
        },
        'model': {
            'name': 'TABS' if method == 'TABS' else 'StandardUNet',
            'class_names': ["Road", "RoadLine", "Vegetation", "Sky", "NoDrivable"],
            'params': {
                'num_classes': 5,
                'initial_filters': 6, # Lightweight 0.27M
                'bilinear': False
            }
        },
        'loss': {
            'name': 'CombinedLoss',
            'params': {'alpha': 0.4, 'beta': 0.6}
        },
        'training': {
            'num_epochs': EPOCHS,
            'batch_size': 8,
            'use_amp': True,
            'early_stopping_patience': 50
        },
        'optimizer': {
            'name': 'AdamW',
            'params': {'lr': 0.001, 'weight_decay': 0.0001}
        },
        'scheduler': {
            'name': 'CosineAnnealingLR',
            'params': {'T_max': EPOCHS, 'eta_min': 0.000001} # Match T_max to epochs
        }
    }
    
    # --- Band Selection Logic ---
    if method == 'TABS':
        config['model']['params']['in_channels'] = 25
        config['model']['params']['num_selected_bands'] = k
    else:
        # Baselines
        if method not in RANKINGS:
            print(f"Error: No rankings for {method}")
            return None
        
        # Take Top-K indices
        indices = RANKINGS[method][:k]
        config['model']['params']['in_channels'] = k # Input dim reduced
        config['model']['params']['selected_bands_indices'] = list(map(int, indices))
        
    return config

def main():
    # Setup directories
    os.makedirs(f'temp_configs_{TARGET_SEED}', exist_ok=True)
    os.makedirs(f'logs_{TARGET_SEED}', exist_ok=True)
    
    processes = []
    
    print(f"🚀 Launching {len(JOBS)} jobs for Seed {TARGET_SEED} (Epochs={EPOCHS})...")
    
    for job in JOBS:
        cfg = create_config(job)
        if not cfg: continue
        
        # Save yaml
        yaml_path = f"temp_configs_{TARGET_SEED}/{cfg['run_name']}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(cfg, f)
            
        # Log file
        log_path = f"logs_{TARGET_SEED}/{cfg['run_name']}.log"
        
        # Command (with PYTHONPATH fix)
        cmd = f"PYTHONPATH=. python scripts/run.py --config {yaml_path} > {log_path} 2>&1"
        
        # Execute
        print(f"  + Started: {cfg['run_name']}")
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)
        
        if len(processes) >= MAX_PARALLEL_JOBS:
            pass

    print(f"\nAll {len(processes)} jobs launched. Waiting for completion...")
    
    # Wait for all
    exit_codes = [p.wait() for p in processes]
    
    print("\n--- SUMMARY ---")
    failures = 0
    for i, code in enumerate(exit_codes):
        if code != 0:
            print(f"❌ Job {JOBS[i]['method']} k={JOBS[i]['k']} FAILED. Check logs_{TARGET_SEED}/")
            failures += 1
    
    if failures == 0:
        print("✅ All jobs completed successfully.")
    else:
        print(f"⚠️  {failures} jobs failed.")

if __name__ == "__main__":
    main()
