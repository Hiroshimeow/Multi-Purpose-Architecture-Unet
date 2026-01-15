# Model Development and Comparison Report

This document provides a comprehensive summary of the iterative development process for the GeminiUNetV2 model. The goal was to create a lightweight model that could achieve a mean Intersection over Union (mIoU) of approximately 0.6 on a test dataset.

## Final Recommendation

After all experiments, the model from **Run 8** is the recommended choice. It provides the best balance of performance and speed, meeting the project's "lightweight" requirement while achieving a respectable mIoU.

*   **Configuration:** `configs/run8_config.yaml`
*   **Best mIoU:** 0.5336
*   **FPS:** ~97
*   **Key Feature:** Uses the small model (32 filters) with learnable transposed convolutions (`bilinear: False`), which provided a significant accuracy boost with no speed penalty.

---

## Experiments Summary Table

| Run | Goal                                      | mIoU                     | FPS                      | Vegetation IoU           | Model Path                                                                 |
|:----|:------------------------------------------|:-------------------------|:-------------------------|:-------------------------|:---------------------------------------------------------------------------|
| 3   | Baseline (Small Model)                    | 0.3980                   | 95.05                    | 0.0433                   | `training_runs/GeminiUNetV2_run1_with_sea...`                              |
| 4   | Increase Model Size                       | 0.5050                   | 47.89                    | 0.1106                   | `training_runs/GeminiUNetV2_run4_filters64...`                             |
| 5   | Small Model + Augmentation                | 0.4473                   | 94.14                    | 0.1370                   | `training_runs/GeminiUNetV2_run5_augmentation...`                          |
| 6   | **Large Model + Augmentation**            | **0.5602**               | 47.96                    | 0.1581                   | `training_runs/GeminiUNetV2_run6_large_augmentation...`                    |
| 7   | Focal Loss Experiment                     | 0.2914                   | 47.88                    | 0.0571                   | `training_runs/GeminiUNetV2_run7_focal_loss...`                            |
| 8   | **Small Model + Transposed Conv**         | **0.5336**               | **96.85**                | **0.1896**               | `training_runs/GeminiUNetV2_run8_small_deconv...`                          |
| 9   | Dice-Focused Loss                         | 0.4995                   | 97.07                    | 0.1947                   | `training_runs/GeminiUNetV2_run9_dice_focused...`                          |

---

## Detailed Experiment Reports

### Run 3: Baseline and SE Attention Ablation

*   **Goal:** Establish a baseline for the small `GeminiUNetV2` model and confirm the effectiveness of the `SE_attention` module.
*   **Configuration Files:** `configs/run3_config.yaml`
*   **Key Parameters:**
    *   **Model:** `GeminiUNetV2`
    *   **Filters:** 32
    *   **Decoder:** Bilinear Upsampling (`bilinear: True`)
    *   **Augmentations:** None
    *   **Loss:** `CombinedLoss` (alpha: 0.6, beta: 0.4)
*   **Experiments:**
    1.  **With SE Attention (`--sea True`):**
        *   **Command:** `python3 run3.py --config configs/run3_config.yaml --epochs 30`
        *   **Result (mIoU):** 0.3980
        *   **Model Path:** `training_runs/GeminiUNetV2_run1_with_sea_20250912_155338`
    2.  **Without SE Attention (`--sea False`):**
        *   **Command:** `python3 run3.py --config configs/run3_config.yaml --epochs 30 --sea False`
        *   **Result (mIoU):** 0.0841
        *   **Model Path:** `training_runs/GeminiUNetV2_run2_no_sea_20250912_155612`
*   **Conclusion:** The `SE_attention` module is critical for model performance.

### Run 4: Increased Model Capacity

*   **Goal:** Measure the impact of increasing model size on accuracy.
*   **Configuration File:** `configs/run4_config.yaml`
*   **Command:** `python3 run4.py --config configs/run4_config.yaml`
*   **Key Parameters:**
    *   **Model:** `GeminiUNetV2`
    *   **Filters:** 64 (Increased from 32)
    *   **Decoder:** Bilinear Upsampling (`bilinear: True`)
    *   **Augmentations:** None
    *   **Loss:** `CombinedLoss` (alpha: 0.6, beta: 0.4)
*   **Results:**
    *   **Best mIoU:** 0.5050
    *   **FPS:** 47.89
*   **Model Path:** `training_runs/GeminiUNetV2_run4_filters64_20250912_155817`
*   **Conclusion:** A larger model improves accuracy but significantly hurts performance, violating the "lightweight" constraint.

### Run 5: Small Model with Data Augmentation

*   **Goal:** Improve the small model's performance without increasing its size by using data augmentation.
*   **Configuration File:** `configs/run5_config.yaml`
*   **Command:** `python3 run5.py --config configs/run5_config.yaml`
*   **Key Parameters:**
    *   **Model:** `GeminiUNetV2` (32 filters)
    *   **Decoder:** Bilinear Upsampling (`bilinear: True`)
    *   **Augmentations:** `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `RandomBrightnessContrast`
    *   **Loss:** `CombinedLoss` (alpha: 0.6, beta: 0.4)
*   **Results:**
    *   **Best mIoU:** 0.4473
    *   **FPS:** 94.14
*   **Model Path:** `training_runs/GeminiUNetV2_run5_augmentation_20250912_160726`
*   **Conclusion:** Augmentation improved the small model's score while maintaining high FPS, but it was not enough to match the large model.

### Run 6: Large Model with Data Augmentation

*   **Goal:** Combine the two most successful strategies to find the maximum potential mIoU of the architecture.
*   **Configuration File:** `configs/run6_config.yaml`
*   **Command:** `python3 run6.py --config configs/run6_config.yaml`
*   **Key Parameters:**
    *   **Model:** `GeminiUNetV2` (64 filters)
    *   **Decoder:** Bilinear Upsampling (`bilinear: True`)
    *   **Augmentations:** `ShiftScaleRotate`, `RandomBrightnessContrast`
    *   **Loss:** `CombinedLoss` (alpha: 0.6, beta: 0.4)
*   **Results:**
    *   **Best mIoU:** 0.5602
    *   **FPS:** 47.96
*   **Model Path:** `training_runs/GeminiUNetV2_run6_large_augmentation_20250912_161207`
*   **Conclusion:** This combination achieved the highest mIoU, but at the cost of low FPS.

### Run 7: Focal Loss Experiment

*   **Goal:** Attempt to solve the poor `Vegetation` class performance by using a specialized loss function.
*   **Configuration File:** `configs/run7_config.yaml`
*   **Command:** `python3 run7.py --config configs/run7_config.yaml`
*   **Key Parameters:**
    *   **Model:** `GeminiUNetV2` (64 filters)
    *   **Augmentations:** `ShiftScaleRotate`, `RandomBrightnessContrast`
    *   **Loss:** `FocalLoss` (alpha: 0.25, gamma: 2.0)
*   **Results:**
    *   **Best mIoU:** 0.2914
    *   **FPS:** 47.88
*   **Model Path:** `training_runs/GeminiUNetV2_run7_focal_loss_20250912_162014`
*   **Conclusion:** The Focal Loss experiment was unsuccessful with the default parameters, destabilizing training and worsening results.

### Run 8: Small Model with Transposed Convolutions (Recommended)

*   **Goal:** Find a compromise between the small and large models by improving the decoder of the small model.
*   **Configuration File:** `configs/run8_config.yaml`
*   **Command:** `python3 run8.py --config configs/run8_config.yaml`
*   **Key Parameters:**
    *   **Model:** `GeminiUNetV2` (32 filters)
    *   **Decoder:** Transposed Convolutions (`bilinear: False`)
    *   **Augmentations:** `ShiftScaleRotate`, `RandomBrightnessContrast`
    *   **Loss:** `CombinedLoss` (alpha: 0.6, beta: 0.4)
*   **Results:**
    *   **Best mIoU:** 0.5336
    *   **FPS:** 96.85
*   **Model Path:** `training_runs/GeminiUNetV2_run8_small_deconv_20250912_162525`
*   **Conclusion:** Highly successful. This configuration provides the best balance of accuracy and speed, making it the recommended model.

### Run 9: Dice-Focused Loss

*   **Goal:** A final attempt to push the Run 8 model over the 0.6 mIoU line by tuning the loss weights.
*   **Configuration File:** `configs/run9_config.yaml`
*   **Command:** `python3 run9.py --config configs/run9_config.yaml`
*   **Key Parameters:**
    *   **Model:** `GeminiUNetV2` (32 filters, `bilinear: False`)
    *   **Augmentations:** `ShiftScaleRotate`, `RandomBrightnessContrast`
    *   **Loss:** `CombinedLoss` (alpha: 0.4, beta: 0.6)
*   **Results:**
    *   **Best mIoU:** 0.4995
    *   **FPS:** 97.07
*   **Model Path:** `training_runs/GeminiUNetV2_run9_dice_focused_20250912_163020`
*   **Conclusion:** Unsuccessful. The change in loss weights hurt overall performance.
