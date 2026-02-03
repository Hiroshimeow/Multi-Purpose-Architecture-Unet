# Project Log - SpectralFormer (SF) Project

## [2026-01-15] - Fully Independent Project Setup
- Migrated HSI Drive 2.1 dataset into `/data1/tabs/SF/Image_dataset/`.
- Updated `configs/spectral_former_base.yaml` and `train.py` to use local relative paths.
- Refined `datasets.py` to correctly locate labels within the internal structure.
- The `SF` project is now 100% self-contained and independent of the external `tabs` files.
- **CRITICAL FIX**: Detected `NaN` loss at Epoch 43 during initial full run.
    - Cause: High Learning Rate (0.0005) for Transformer architecture.
    - Solution: 
        1. Reduced LR to 0.0001 in `configs/spectral_former_base.yaml`.
        2. Increased Batch Size to 16 for gradient stability.
        3. Implemented **Gradient Clipping (max_norm=1.0)** in `SF/src/trainer.py`.
- Restarted full training (400 epochs) with enhanced stability settings.
- **[2026-01-15] - Ceiling Search Initiated**:
    - Launched `SF-Huge-P16` (dim=512, depth=12) to test massive capacity.
    - Launched `SF-Base-P8` (patch_size=8) to test high-resolution attention.
- **[2026-01-15] - P8 Architecture Fix**:
    - Identified `RuntimeError` in P8 due to hard-coded upsampling layers in `SpectralFormer_Seg`.
    - Modified `SF/models/spectral_former_seg.py` to calculate `num_upsamples` dynamically based on `log2(patch_size)`.
    - **FIX**: Added missing `import numpy as np` in `spectral_former_seg.py` after `NameError`.
    - Relaunched `SF-Base-P8` experiment.
    - Batch size set to 8 for these runs to manage 24GB VRAM efficiently.
