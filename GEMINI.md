# Project Context

- **Date:** 2025年9月12日金曜日
- **OS:** linux
- **Working Directory:** /data1/hai/unet/Multi-Purpose-Architecture-Unet

## Folder Structure

```
/data1/hai/unet/Multi-Purpose-Architecture-Unet/
├───.gitignore
├───analyze.py
├───README.md
├───requirements.txt
├───run.py
├───tất_cả_mã_nguồn.txt
├───unet_hsi.py
├───unet_tool.py
├───.git/...
├───.stfolder/...
├───configs/...
├───dataset_40_for_test/...
├───Image_dataset/...
├───run_full_manguon/...
├───src/...
└───training_runs/...
```

## Experiment History

*(Lưu ý cho các phiên làm việc sau: Luôn luôn ghi lại tóm tắt của mỗi lần chạy thử nghiệm vào phần này để duy trì một bản ghi dự án hoàn chỉnh.)*

### Run 3: GeminiUNetV2 with and without SE_attention

*   **Goal:** Achieve 0.7 mIoU with a lightweight model, comparing performance with and without the `SE_attention` module.
*   **Model:** `GeminiUNetV2` with `SAC` block.
*   **Epochs:** 30

**Results:**

1.  **With SE_attention (`--sea True`):**
    *   **Best mIoU:** 0.3980
    *   **FPS:** 95.05
    *   **Conclusion:** The model trained for all 30 epochs but did not reach the 0.7 mIoU target. Performance is much better than without SE_attention.

2.  **Without SE_attention (`--sea False`):**
    *   **Best mIoU:** 0.0841
    *   **FPS:** 104.67
    *   **Conclusion:** The model failed to learn, and training stopped early.

**Overall Conclusion:** The `SE_attention` module is critical for performance. The current model architecture is too small to achieve the target mIoU of 0.7 within 30 epochs. The next step is to increase model capacity and training time.

### Run 4: Increased Model Capacity

*   **Goal:** Improve mIoU by increasing model size.
*   **Model:** `GeminiUNetV2` with `initial_filters` and `sac_params.channels` increased from 32 to 64.
*   **Epochs:** 50

**Results:**

*   **Best mIoU:** 0.5050
*   **FPS:** 47.89
*   **Latency:** 167.05 ms
*   **Conclusion:** Increasing model capacity improved mIoU significantly (from 0.3980 to 0.5050), but it is still below the 0.7 target. The larger model is much slower, hurting the "lightweight" goal. The model still struggles with the "Vegetation" class.

**Overall Conclusion:** Simply increasing model size is not the optimal path. The next step is to try and improve the smaller model's performance using data augmentation.

### Run 5: Small Model with Data Augmentation

*   **Goal:** Improve the lightweight model's mIoU using data augmentation.
*   **Model:** `GeminiUNetV2` with `initial_filters: 32`.
*   **Augmentations:** `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `RandomBrightnessContrast`.
*   **Epochs:** 50

**Results:**

*   **Best mIoU:** 0.4473
*   **FPS:** 94.14
*   **Conclusion:** Data augmentation improved the mIoU of the small model from 0.3980 to 0.4473 while maintaining high FPS. However, this was not enough to surpass the performance of the larger model (Run 4).

**Overall Conclusion:** Data augmentation is a valuable tool, but the model still appears to be capacity-bound. The next logical step is to combine a larger model with data augmentation to test the architecture's maximum potential.

### Run 6: Large Model with Data Augmentation

*   **Goal:** Combine the large model with data augmentation to maximize mIoU.
*   **Model:** `GeminiUNetV2` with `initial_filters: 64`.
*   **Augmentations:** `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `RandomBrightnessContrast`.
*   **Epochs:** 50

**Results:**

*   **Best mIoU:** 0.5602
*   **FPS:** 47.96
*   **Conclusion:** Combining the large model with data augmentation yielded the best mIoU so far, confirming that both strategies are additive. However, the result is still significantly below the 0.7 target.

**Overall Conclusion:** We have likely reached the performance limit of the current architecture with the standard loss function. The primary issue remains the very poor performance on the `Vegetation` class (IoU of 0.1581). The next step is to try a more advanced loss function, Focal Loss, to specifically address this class imbalance.

### Run 7: Large Model with Focal Loss

*   **Goal:** Address class imbalance using Focal Loss to improve the mIoU, specifically for the `Vegetation` class.
*   **Model:** `GeminiUNetV2` with `initial_filters: 64`.
*   **Augmentations:** `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `RandomBrightnessContrast`.
*   **Loss Function:** `FocalLoss` (alpha=0.25, gamma=2.0).
*   **Epochs:** 50

**Results:**

*   **Best mIoU:** 0.2914
*   **FPS:** 47.88
*   **Conclusion:** The experiment with Focal Loss was unsuccessful. The mIoU dropped significantly from 0.5602 to 0.2914, and training was unstable. The IoU for the `Vegetation` class worsened, and other classes also saw a major performance degradation.

### Run 8: Small Model with Transposed Convolutions

*   **Goal:** Improve the small model's mIoU by using learnable transposed convolutions in the decoder.
*   **Model:** `GeminiUNetV2` with `initial_filters: 32` and `bilinear: False`.
*   **Augmentations:** `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `RandomBrightnessContrast`.
*   **Epochs:** 50

**Results:**

*   **Best mIoU:** 0.5336
*   **FPS:** 96.85
*   **Conclusion:** This was a highly successful experiment. Using transposed convolutions significantly boosted the lightweight model's mIoU from 0.4473 (Run 5) to 0.5336, with no performance penalty. This model represents the best balance of speed and accuracy so far.

**Overall Conclusion:** The model from Run 8 is the best lightweight candidate, coming close to the 0.6 mIoU target while maintaining high FPS. The final step will be to tune the loss function weights to give more emphasis to the Dice component, which may push the IoU score over the finish line.

### Run 9: Dice-Focused Loss

*   **Goal:** Tweak the loss function weights to favor the Dice score and potentially pass the 0.6 mIoU threshold.
*   **Model:** `GeminiUNetV2` with `initial_filters: 32` and `bilinear: False`.
*   **Loss Weights:** `alpha: 0.4`, `beta: 0.6`.
*   **Epochs:** 50

**Results:**

*   **Best mIoU:** 0.4995
*   **FPS:** 97.07
*   **Conclusion:** This final adjustment was not successful. Increasing the weight of the Dice loss component actually decreased the overall mIoU compared to Run 8.

## Final Project Summary

After an extensive series of experiments, we have determined that the optimal model configuration that balances performance and the "lightweight" requirement is the one from **Run 8**.

**Recommended Model:**
*   **Configuration:** `configs/run8_config.yaml`
*   **Architecture:** Small model (`initial_filters: 32`) with the `SAC` attention block and learnable transposed convolutions (`bilinear: False`).
*   **Training:** Trained with data augmentation.
*   **Performance:**
    *   **Best mIoU:** 0.5336
    *   **FPS:** ~97
    *   **Training Log:** Stable and healthy convergence.

This model represents the best outcome of our iterative development process. It is fast, reasonably accurate, and provides a strong foundation for future work, even though it did not ultimately reach the 0.6 mIoU target. The primary remaining challenge is improving the detection of the "Vegetation" class, which would likely require architectural changes beyond the scope of this investigation.
