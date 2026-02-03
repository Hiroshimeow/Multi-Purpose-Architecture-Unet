# Experimental Results Comparison

## Dataset: HSI Drive 2.1

| Model | Bands | k | Params (M) | GFLOPs | mIoU (Best) | FPS |
|-------|-------|---|------------|--------|-------------|-----|
| TABS-UNet (Ref) | 25 | 5 | 0.27 | 1.02 | 0.7811 | ~ |
| **SpectralFormer-Seg (Base)** | 25 | - | 2.15 | 1.01 | *Training...* | 1170.99 |
| **SpectralFormer-Seg (Huge)** | 25 | - | 42.31 | 17.34 | *Training...* | ~ |
| **SpectralFormer-Seg (P8)** | 25 | - | *Pending* | *Pending* | *Waiting* | *Pending* |

*(Note: TABS-UNet results are for reference from previous best run)*
