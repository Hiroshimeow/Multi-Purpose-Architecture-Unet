# Development Plan - SpectralFormer for Semantic Segmentation (ADS)

## 1. Objective
- Adapt the SpectralFormer (originally for pixel classification) into a full semantic segmentation model.
- Compare performance with TABS-UNet on HSI Drive 2.1 dataset.
- Focus on lightweight execution suitable for embedded ADS hardware.

## 2. Architecture: SpectralFormer-Seg
- **Encoder**: Use SpectralFormer's Transformer blocks with:
    - **GSE (Group-wise Spectral Embedding)**: Better spectral feature extraction.
    - **CAF (Cross-layer Adaptive Fusion)**: Multi-scale spectral-spatial fusion.
- **Decoder**: Convolutional upsampling (similar to UNet decoder) to project latent features back to original resolution (H, W).
- **Task**: Semantic Segmentation (5 classes).

## 3. Data Strategy
- **Dataset**: HSI Drive 2.1 (Matlab version).
- **Input**: `cubes_fl32/Cubes_Scaling/` (25 bands).
- **Labels**: `labels/`.
- **Patching**: 224x224 (adjustable for efficiency).

## 5. Ceiling Search Strategy (High Capacity Experiments)
- Aim: Identify the maximum achievable mIoU using full 25 bands without resource constraints.
- Experiments:
    - **SF-Huge**: Increased `dim` (512), `depth` (12), and `heads` (16) for massive learning capacity.
    - **SF-Granular**: Reduced `patch_size` to 8 for higher spatial resolution processing in Transformer.
    - Monitor VRAM and stability (Gradient Clipping enabled).
