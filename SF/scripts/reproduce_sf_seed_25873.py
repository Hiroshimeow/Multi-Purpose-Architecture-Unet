
import os
import yaml
import subprocess
import time
import sys
from pathlib import Path

# --- CONFIGURATION ---
TARGET_SEED = 25873
EPOCHS = 400
MAX_PARALLEL_JOBS = 20
BASE_OUTPUT_DIR = './training_runs_reproduce'

# Rankings from calculate_bands.py (Best to Worst) - Identical to TABS
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
    
    run_name = f"SF_S{seed}_{method}_k{k}_Tiny"
    
    # Tiny Architecture (0.46M params with P16)
    model_params = {
        'image_size': 224,
        'patch_size': 16, # Critical for low params
        'in_channels': k if method != 'TABS' else 25, # TABS takes full input
        'num_classes': 5,
        'dim': 64,        # Tiny
        'depth': 3,       # Tiny
        'heads': 4,
        'mlp_dim': 128,   # Tiny
        'dim_head': 16,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }
    
    if method == 'TABS':
        model_params['num_selected_bands'] = k

    config = {
        'run_name': run_name,
        'base_output_dir': BASE_OUTPUT_DIR,
        'data': {
            'dir': "../Image_dataset/cubes_fl32/", # Corrected path
            'file_extension': 'mat',
            'normalization': 'min-max',
            'class_mapping': {1: 0, 2: 1, 3: 2, 5: 3, 0: 4, 4: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4},
            'patching': {'patch_size': 224, 'stride': 192, 'use_patching': True},
            'seed': seed 
        },
        'augmentation': {
            'use_augmentation': True,
            'shift_scale_rotate': {'p': 0.5, 'shift_limit': 0.0625, 'scale_limit': 0.1, 'rotate_limit': 15},
            'random_brightness_contrast': {'p': 0.5}
        },
        'model': {
            'name': 'SpectralFormer_Seg',
            'class_names': ["Road", "RoadLine", "Vegetation", "Sky", "NoDrivable"],
            'params': model_params
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
            'params': {'lr': 0.0005, 'weight_decay': 0.0001} # Higher LR for smaller model (5e-4)
        },
        'scheduler': {
            'name': 'CosineAnnealingLR',
            'params': {'T_max': EPOCHS, 'eta_min': 1e-6}
        }
    }
    
    # --- Band Selection Logic ---
    if method == 'TABS':
        pass # TABS learns internally
    else:
        if method not in RANKINGS:
            print(f"Error: No rankings for {method}")
            return None
        
        # Take Top-K indices
        indices = RANKINGS[method][:k]
        # Inject indices so Dataset slices them
        config['model']['params']['selected_bands_indices'] = list(map(int, indices))
        
    return config

def main():
    # Setup directories
    os.makedirs('temp_configs_reproduce', exist_ok=True)
    os.makedirs('logs_reproduce', exist_ok=True)
    
    processes = []
    
    print(f"🚀 Launching {len(JOBS)} SF-Tiny jobs for Seed {TARGET_SEED}...")
    
    for job in JOBS:
        cfg = create_config(job)
        if not cfg: continue
        
        # Save yaml
        yaml_path = f"temp_configs_reproduce/{cfg['run_name']}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(cfg, f)
            
        # Log file
        log_path = f"logs_reproduce/{cfg['run_name']}.log"
        
        # Command
        # Need to ensure python path includes current dir
        cmd = f"PYTHONPATH=. python train.py --config {yaml_path} > {log_path} 2>&1"
        
        # Execute
        print(f"  + Started: {cfg['run_name']}")
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)
        
        # Rate limit
        if len(processes) >= MAX_PARALLEL_JOBS:
             pass # We only have 15 jobs, so this is fine

    print(f"\nAll {len(processes)} jobs launched. Waiting for completion...")
    
    # Wait for all
    exit_codes = [p.wait() for p in processes]
    
    print("\n--- SUMMARY ---")
    failures = 0
    for i, code in enumerate(exit_codes):
        if code != 0:
            print(f"❌ Job {JOBS[i]['method']} k={JOBS[i]['k']} FAILED. Check logs_reproduce/")
            failures += 1
    
    if failures == 0:
        print("✅ All jobs completed successfully.")
    else:
        print(f"⚠️  {failures} jobs failed.")

if __name__ == "__main__":
    main()
