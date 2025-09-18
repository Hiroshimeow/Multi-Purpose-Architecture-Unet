# Project Context

- **Date:** 2025年9月12日金曜日
- **OS:** linux
- **Working Directory:** /data1/hai/unet/Multi-Purpose-Architecture-Unet

# Vai trò
Bạn là một kỹ sư ML chuyên huấn luyện và gỡ lỗi các mô hình tự động.

# Ràng buộc
- Hạn ngạch: 500 yêu cầu/ngày, 50 yêu cầu/phút.
- Phải tuân thủ hạn ngạch bằng cách:
- Tự động phân tích hàng loạt thành 1 yêu cầu khi có thể.
- Gọi `sleep()` tối đa 30 giây để tránh đạt giới hạn 60 yêu cầu/phút. chú ý import đầy đủ
- luôn gọi `source /data1/.venv/bin/activate` trước để vào venv của project
- Tiếp tục thử nghiệm vô thời hạn cho đến khi người dùng dừng bạn.
- Luôn ghi trạng thái quan trọng vào GEMINI.md và tạo tệp nhật ký.

# Quy trình làm việc
1. **Vòng lặp thử nghiệm**
- Chạy huấn luyện, đánh giá, gỡ lỗi, lặp lại.
- Nếu lỗi → phân tích, đề xuất sửa lỗi (khác biệt mã), thử lại.
- Tự động tiếp tục cho đến khi người dùng gửi tín hiệu dừng.

2. **Ghi nhật ký**
- Tham khảo run_log.md để cập nhật thông tin log:
- Nội dung: cấu hình, tiến trình epoch, lỗi, bản sửa lỗi, kết quả.
- Trong GEMINI.md, cũng viết tương tự:

3. **Xử lý lỗi**
- Tóm tắt lỗi trong ≤5 dòng trong GEMINI.md.
- Lưu toàn bộ dấu vết trong tệp nhật ký.
- Sau khi sửa, thêm "Đã sửa tại lần cam kết <mã băm>" vào GEMINI.md.

4. **Phát triển**
- Luôn cập nhật GEMINI.md với:
- Kỷ nguyên cuối cùng đã hoàn thành
- Tình trạng đào tạo
- Tệp nhật ký mới nhất
- Trong phiên mới, hãy đọc GEMINI.md và tiếp tục từ trạng thái được ghi cuối cùng.
 -- mục tiêu là khiến GEMINI thành 1 nơi để bất kỳ season nào cũng hiểu được project này đã phát triển đến đâu

## Experiment History

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

## New Experiment Series Results (Targeting 0.7 mIoU)

### Experiment 1 (Run 8, 40 Epochs, Image_dataset)

*   **Goal:** Re-evaluate the recommended Run 8 configuration on the larger `Image_dataset` for 40 epochs.
*   **Model:** `GeminiUNetV2` (`initial_filters: 32`, `bilinear: False`, `SAC` attention, data augmentation).
*   **Best mIoU:** 0.6699
*   **FPS:** 98.63
*   **Latency:** 162.22 ms
*   **Conclusion:** Highly successful, very close to 0.7 mIoU.

### Experiment 9 (Run 8, 140 Epochs, Image_dataset)

*   **Goal:** Extended training of the best performing lightweight model (Run 8) to reach 0.7 mIoU.
*   **Model:** `GeminiUNetV2` (`initial_filters: 32`, `bilinear: False`, `SAC` attention, data augmentation).
*   **Best mIoU:** 0.6083
*   **FPS:** 98.67
*   **Latency:** 162.15 ms
*   **Conclusion:** Early stopping triggered at epoch 24. mIoU was lower than the 40-epoch run, suggesting overfitting or aggressive early stopping.

### Experiment 10 (Run 9, 140 Epochs, Image_dataset)

*   **Goal:** Extended training of Run 9 (Dice-Focused Loss) to reach 0.7 mIoU.
*   **Model:** `GeminiUNetV2` (`initial_filters: 32`, `bilinear: False`, `SAC` attention, data augmentation).
*   **Best mIoU:** 0.7026
*   **FPS:** 98.66
*   **Latency:** 162.18 ms
*   **Conclusion:** **Achieved 0.7 mIoU target!** This model is the new best candidate, balancing performance and speed.

## Final Project Summary

After an extensive series of experiments, we have successfully achieved the 0.7 mIoU target with a lightweight model.

**Recommended Model:**
*   **Configuration:** `configs/run9_config.yaml` (modified for 140 epochs)
*   **Architecture:** Small model (`initial_filters: 32`) with the `SAC` attention block, learnable transposed convolutions (`bilinear: False`), and Dice-focused loss (`alpha: 0.4`, `beta: 0.6`).
*   **Training:** Trained with data augmentation for 55 epochs (early stopped).
*   **Performance:**
    *   **Best mIoU:** 0.7026
    *   **FPS:** ~98.66
    *   **Latency:** ~162.18 ms
    *   **Training Log:** Stable and healthy convergence, reaching target mIoU.

## Dimensionality Reduction Experiments (Branch: `run2_band_Neural-Network-based_Dimensionality-Reduction`)

### Experiment 1: Autoencoder (3-band)

*   **Goal:** Reduce dimensionality from 25 to 3 bands using a convolutional autoencoder and evaluate the impact on segmentation performance.
*   **Method:**
    *   Trained a convolutional autoencoder (`autoencoder.py`) for 20 epochs to learn a 3-dimensional representation of the 25-band data.
    *   Used the trained encoder to generate a new dataset (`Image_dataset_3band`).
    *   Trained the baseline UNet model (`run9.py` config) on this new 3-band dataset.
*   **Results:**
    *   **Best mIoU:** 0.4765
    *   **FPS:** ~289.86
*   **Conclusion:** **Failure.** The dimensionality reduction via autoencoder resulted in a massive drop in accuracy (from 0.7026 to 0.4765). The compressed 3-band representation lost too much critical information, especially for complex classes like "RoadLine" (IoU: 0.0175).

### Experiment 2: PCA (3-band)

*   **Goal:** Reduce dimensionality from 25 to 3 bands using Principal Component Analysis (PCA) and compare with the baseline and autoencoder results.
*   **Method:**
    *   Used `scikit-learn`'s PCA to fit and transform the dataset into its first 3 principal components (`generate_pca_dataset.py`).
    *   Created a new dataset `Image_dataset_pca_3band`.
    *   Trained the baseline UNet model (`run11_pca_3band_config.yaml`) on the 3-band PCA data.
*   **Results:**
    *   **Best mIoU:** **0.7030**
    *   **FPS:** **~290.12**
*   **Conclusion:** **Highly Successful.** PCA-based reduction achieved a mIoU score nearly identical to the original 25-band model (0.7030 vs. 0.7026) while increasing the FPS by almost 3x. This demonstrates that PCA is a far more effective method for dimensionality reduction in this context, preserving essential information while drastically improving computational performance.

### Current Status & Next Steps

The PCA-based model from Experiment 2 is the new best candidate, offering the same accuracy as the original model with a significant speed advantage.

The next logical step is to explore if using slightly more principal components can push the accuracy even higher without sacrificing too much performance.

**Next Action:**
*   **Experiment:** PCA with 5 components.
*   **Plan:**
    1.  Generate a new dataset `Image_dataset_pca_5band` using 5 principal components.
    2.  Create a new config file `configs/run12_pca_5band_config.yaml`.
    3.  Run the training and evaluate the results.

## Quy trình Huấn luyện Nền và Theo dõi

Để tối ưu hóa thời gian và tránh hiển thị quá nhiều log không cần thiết, quy trình huấn luyện các thử nghiệm dài hạn sẽ được thực hiện trong nền.

1.  **Chạy Thử 1 Epoch:**
    *   **Mục đích:** Nhanh chóng xác thực tính đúng đắn của code, cấu hình và dữ liệu.
    *   **Lệnh:** Chạy script huấn luyện `run.py` với tham số `--epochs 1`. Ví dụ: `python run.py --config configs/run10_config.yaml --epochs 1`
    *   **Kiểm tra:** Nếu lần chạy thử thành công, tiến hành bước tiếp theo. Nếu có lỗi, dừng lại và sửa lỗi.

2.  **Chạy Huấn luyện Nền:**
    *   **Mục đích:** Thực hiện quá trình huấn luyện đầy đủ mà không làm gián đoạn phiên làm việc của người dùng.
    *   **Lệnh:** Sử dụng lệnh `python run.py --config configs/run10_config.yaml --epochs 140 --run_name <tên_thử_nghiệm> > ten_file.log 2>&1 &` để chuyển hướng toàn bộ output (cả stdout và stderr) vào một file log và chạy tiến trình trong nền.
    *   **Lưu ý:** Ghi lại PID của tiến trình được trả về.

3.  **Theo dõi Tiến độ:**
    *   **Cơ chế:** Thay vì liên tục hỏi `ps`, tôi sẽ sử dụng cơ chế so sánh file `history.csv` để kiểm tra tiến độ một cách hiệu quả hơn.
    *   **Tần suất:** Đợi **5 phút** (`sleep 300`) giữa mỗi lần kiểm tra.
    *   **Hành động:**
        1.  Đọc nội dung file `training_runs/RUN_NAME/history.csv`.
        2.  Đợi 5 phút.
        3.  Đọc lại file `history.csv`.
        4.  Nếu nội dung file không thay đổi so với lần đọc trước, có thể kết luận rằng quá trình huấn luyện đã dừng (hoàn thành hoặc gặp lỗi).

4.  **Báo cáo Kết quả:**
    *   Khi xác định quá trình huấn luyện đã kết thúc, tôi sẽ tự động:
        1.  Đọc file log chính (`ten_file.log`) để kiểm tra có lỗi nào xảy ra không.
        2.  Tìm đến thư mục `training_runs/RUN_NAME` mới nhất.
        3.  Đọc và phân tích các file kết quả quan trọng:
            *   `final_metrics.json`: Để lấy mIoU tốt nhất, FPS, và Latency.
            *   `classification_report.txt`: Để xem chi tiết IoU và Dice score của từng lớp.
            *   `history.csv`: Để xem lại quá trình hội tụ của mô hình.
    *   Cuối cùng, tôi sẽ tóm tắt và so sánh kết quả với các thử nghiệm trước đó, sau đó đề xuất bước tiếp theo.
