# Multi-Purpose U-Net Training Framework

This repository provides a structured and modular framework for training U-Net architectures, specifically designed for multi-channel data and flexible experimentation.

## 🚀 Quick Start

### Environment Setup
The project is optimized for **NVIDIA RTX A5000** GPUs.
```bash
# Activate the virtual environment
source /data1/.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Common Execution Commands
```bash
# Basic training with a custom run name
python run.py --config configs/unet_stable_loss.yaml --run_name tangpatchsize

# Training with a specific config and scheduler name
python run.py --config configs/unet_paper_replica.yaml --run_name unet_StepLR

# Analyze a single completed run
python analyze.py training_runs/GeminiUNet_Run_1_20250912_122252

# Compare multiple runs
python analyze.py training_runs/unet_run_A_20250911_140000 training_runs/unet_run_B_20250911_150000
```

---

## 📂 Project Structure

```text
Multi-Purpose-Architecture-Unet/
├── Image_dataset/              # Data storage
│   ├── cubes_fl32/             # Input tensors/images
│   └── labels/                 # Ground truth masks
├── configs/
│   └── unet_base.yaml          # Experiment configuration files
├── src/
│   ├── __init__.py
│   ├── datasets.py             # Dataset class definitions
│   ├── losses.py               # Loss function definitions
│   ├── models.py               # Model architecture definitions
│   ├── experiment_manager.py   # File, log, and checkpoint management
│   ├── plotter.py              # Visualization and plotting utilities
│   └── trainer.py              # Training loop logic
├── run.py                      # Unified entry point for training
└── analyze.py                  # Script for post-training analysis
```

---

## 🔬 Detailed Component Analysis

### `run.py` - The Conductor
The single entry point of the project. It handles command-line arguments (`argparse`), loads/merges YAML configurations, and initializes the `ExperimentManager`, `Dataset`, `Model`, and `Trainer`.

### `configs/` - The Blueprint
Defines the entire experiment setup, including hyperparameters, paths, and component names (model, loss, optimizer). This ensures a clean separation between configuration and source code.

### `src/trainer.py` - The Engine
Encapsulates the training loop logic. It manages epochs, calls training/evaluation steps, triggers the `ExperimentManager` for logging, handles Early Stopping, and manages state restoration during resumes.

### `src/experiment_manager.py` - The Manager
Handles all file system operations and experiment states:
- Creates unique result directories.
- Saves final merged configurations.
- Logs metrics to `.csv` files.
- Saves/loads checkpoints (model, optimizer, scheduler).
- Generates final summary reports.

### `src/models.py` - The Model Factory
Contains definitions for all model architectures. It includes a `get_model(name, params)` factory function to instantiate models dynamically based on the configuration file.

### `src/datasets.py` & `src/losses.py` - Component Libraries
Contain reusable `Dataset` and `Loss` classes, providing the fundamental building blocks for the training process.

### `src/plotter.py` - The Artist
An independent module focused on visualization: training history (loss, mIoU), confusion matrices, and per-class metric charts.

---

## 📖 Usage Guide

### 1. Basic Experiment
Run a training session using the default configuration. The framework automatically appends a timestamp to the run name to prevent overwriting.
```bash
python run.py --run_name unet_baseline
```

### 2. Custom Configuration
If you want to test a specific architecture (e.g., a "Slim" U-Net with fewer filters), create a new config file and specify it:
```bash
python run.py --config configs/unet_slim.yaml --run_name unet_slim_32filters
```

### 3. Hyperparameter Overrides
Useful for quick tuning without modifying YAML files. This command overrides the learning rate and batch size from the config.
```bash
python run.py --run_name unet_lr_1e-3_bs_8 --lr 0.001 --batch_size 8
```

### 4. Resume Training
If a session is interrupted, you can resume exactly where it left off by pointing to the directory of the run. The framework loads the weights, optimizer state, and epoch count.
```bash
python run.py --resume_path training_runs/unet_lr_1e-3_bs_8_20250910_141500
```

### 5. Post-Training Analysis
Use `analyze.py` to regenerate plots or view summaries of a completed run without re-running the model.
```bash
python analyze.py training_runs/unet_baseline_20250910_140000
```

### 6. Adjusting Input Channels
To experiment with different input dimensions (e.g., 15 channels instead of 25), simply update the `in_channels` parameter in your config file:
```yaml
# configs/unet_15channels.yaml
model:
  name: "UNet"
  params:
    in_channels: 15
    num_classes: 5
```
Then run:
```bash
python run.py --config configs/unet_15channels.yaml --run_name unet_15channels
```
