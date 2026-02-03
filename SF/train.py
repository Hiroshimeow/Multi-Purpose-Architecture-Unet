import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np

# Ensure local SF modules are prioritized
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

from src.datasets import TiledHyperspectralDataset
from src.trainer import Trainer
from src.experiment_manager import ExperimentManager
from src.losses import CombinedLoss
from models import get_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config_path, epochs_override=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if epochs_override:
        config['training']['num_epochs'] = epochs_override
        print(f"Overriding epochs to {epochs_override}")

    set_seed(config['data']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading - handle relative path within SF
    data_dir_cfg = config['data']['dir']
    if not os.path.isabs(data_dir_cfg):
        data_dir = str(current_dir / data_dir_cfg)
    else:
        data_dir = data_dir_cfg
    
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(config['data']['file_extension'])])
    
    # Split 60/20/20
    random.shuffle(all_files)
    n = len(all_files)
    train_files = all_files[:int(n*0.6)]
    val_files = all_files[int(n*0.6):int(n*0.8)]
    test_files = all_files[int(n*0.8):]
    
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    train_dataset = TiledHyperspectralDataset(train_files, config, augment=True)
    val_dataset = TiledHyperspectralDataset(val_files, config, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = get_model(config).to(device)
    
    # Criterion
    if config['loss']['name'] == 'CombinedLoss':
        criterion = CombinedLoss(**config['loss']['params'])
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer']['params'])
    
    if config['scheduler']['name'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['scheduler']['params'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Experiment Manager
    run_dir, is_new = ExperimentManager.setup_run_directory(config['base_output_dir'], config['run_name'])
    manager = ExperimentManager(run_dir, config['model']['class_names'], is_new_run=is_new)
    manager.save_config(config)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        manager=manager,
        device=device,
        config=config,
        num_epochs=config['training']['num_epochs']
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/spectral_former_base.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    main(args.config, args.epochs)
