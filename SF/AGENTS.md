# AI Agent Guide - SpectralFormer

This guide is intended for AI coding agents (like Cursor, Copilot, or Opencode) working on the SpectralFormer repository. It outlines the project structure, development workflows, and coding standards.

## 1. Project Overview
SpectralFormer is a research project for hyperspectral image (HSI) classification using Transformer-based architectures. It implements group-wise spectral embedding (GSE) and cross-layer adaptive fusion (CAF) for both pixel-wise and patch-wise classification.

## 2. Environment & Dependencies
- **Language**: Python 3.7+
- **Core Frameworks**: PyTorch 1.6+, NumPy, SciPy, Scikit-learn
- **Hardware**: GPU with CUDA 10.1+ recommended.
- **Data Format**: `.mat` files (Matlab) for HSI data and labels.

## 3. Core Commands

### 3.1 Training & Evaluation (Modern Workflow)
The project uses YAML configurations for modular experiments.
- **Train a model**:
  ```bash
  python train.py --config configs/spectral_former_base.yaml --epochs 300
  ```
- **Override epochs**:
  ```bash
  python train.py --config configs/sf_base_p8.yaml --epochs 100
  ```

### 3.2 Demo & Baseline Workflow (Legacy/Research)
`demo.py` is used for quick experiments and replicating paper results.
- **Train ViT (Indian Pines)**:
  ```bash
  python demo.py --dataset='Indian' --epoch=1400 --patches=1 --band_patches=1 --mode='ViT' --weight_decay=0
  ```
- **Train SpectralFormer (Pixel-wise)**:
  ```bash
  python demo.py --dataset='Indian' --epoches=290 --patches=1 --band_patches=3 --mode='CAF' --weight_decay=0
  ```
- **Test/Evaluate a saved model**:
  ```bash
  python demo.py --dataset='Indian' --flag_test=test --patches=1 --band_patches=3 --mode='CAF'
  ```

### 3.3 Running Tests
There is no formal unit test suite (e.g., pytest). "Testing" refers to evaluating model performance on the HSI datasets.
- To verify a code change doesn't break training, run a short training session:
  ```bash
  python train.py --config configs/spectral_former_base.yaml --epochs 1
  ```

## 4. Project Structure
- `models/`: Architecture definitions (e.g., `SpectralFormer_Seg`, `ViT`).
- `src/`: Core logic and utilities.
    - `datasets.py`: PyTorch dataset implementations for HSI data.
    - `trainer.py`: High-level training and evaluation loops.
    - `experiment_manager.py`: Handles logging, checkpoints, and result plotting.
    - `utils.py`: Helper functions for data processing.
- `configs/`: YAML files defining model hyperparameters and training settings.
- `data/` & `Image_dataset/`: Directories for raw and processed hyperspectral data.
- `logs/` & `training_runs/`: Output directories for checkpoints and metrics.

## 5. Coding Style Guidelines

### 5.1 Python Standards
- Follow **PEP 8** for general formatting.
- **Naming Conventions**:
    - Classes: `PascalCase` (e.g., `SpectralFormer_Seg`).
    - Functions/Variables: `snake_case` (e.g., `calculate_miou`).
    - Constants: `UPPER_SNAKE_CASE`.
- **Typing**: Use type hints for all new function signatures (e.g., `def train(self, num_epochs: int) -> None:`).

### 5.2 Imports
Group imports in the following order:
1. Standard library (os, sys, time, etc.)
2. Third-party libraries (torch, numpy, scipy)
3. Local modules (src.trainer, models.get_model)

```python
import os
import torch
import numpy as np
from src.trainer import Trainer
```

### 5.3 Error Handling
- Use `try...except...finally` blocks for hardware-related operations (CUDA) and File I/O.
- Ensure CUDA resources are managed safely; see `src/trainer.py` for cuDNN initialization safety.

### 5.4 Documentation
- Use docstrings for classes and public methods.
- Keep comments concise and in English. Avoid adding redundant comments that describe *what* the code does; focus on *why*.

## 6. AI Agent Best Practices

### 6.1 Modifying Models
- When adding new layers or modules, place them in `models/base_blocks.py` if they are reusable.
- Update `models/__init__.py:get_model` when adding new model classes.

### 6.2 Data Handling
- Hyperspectral data is high-dimensional. Always check tensor shapes before applying transformations.
- Most data processing is done in `src/utils.py` (mirroring, normalization) and `src/datasets.py`.

### 6.3 Configuration
- Prefer adding parameters to YAML configs in `configs/` rather than hardcoding them in `train.py`.
- Use `ExperimentManager` to ensure all changes are logged in the `training_runs/` directory.

### 6.4 Risk Analysis (Vietnamese Instruction Compliance)
Khi thực hiện thay đổi, hãy phân tích rủi ro liên quan đến:
1. **Memory**: Dữ liệu HSI rất lớn, dễ gây lỗi Out of Memory (OOM).
2. **CUDA**: Đảm bảo các tensor nằm trên cùng một thiết bị (device).
3. **Compatibility**: Giữ nguyên tính tương thích với định dạng `.mat` hiện tại.
4. **Context**: Nếu thiếu thông tin về dataset hoặc config, hãy báo "không đủ thông tin".

## 7. Known Issues & Workarounds
- **cuDNN Initialization**: If you encounter `CUDNN_STATUS_NOT_INITIALIZED`, ensure `torch.backends.cudnn.enabled = False` is set as seen in `src/trainer.py`.
- **Memory Errors**: For large datasets like Pavia or Houston, use smaller patch sizes or split images into sub-images during testing.

---
*This file is a living document. Update it as project conventions evolve.*
