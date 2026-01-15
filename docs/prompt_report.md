# Báo cáo Tiến độ Giữa Kỳ: Tối ưu hóa Mô hình Phân vùng Ảnh Viễn thám

## Slide 1: Giới thiệu

*   **Tiêu đề:** Báo cáo Tiến độ Giữa Kỳ: Tối ưu hóa Mô hình Phân vùng Ảnh Viễn thám
*   **Mục tiêu tổng thể:** Xây dựng mô hình lightweight cho phân vùng ảnh viễn thám đa kênh (25 bands), đạt mIoU > 0.7, FPS cao, và tối ưu hóa GPU.

## Slide 2: UNet-base: Điểm Khởi Đầu Mạnh Mẽ

*   **Mô tả:** Kiến trúc UNet cơ bản làm baseline.
*   **Kết quả:** Đạt mIoU **0.8232**.
*   **Ý nghĩa:** Cho thấy khả năng phân vùng tốt ngay từ đầu.
*   **Gợi ý hình ảnh:** Sơ đồ khối đơn giản của kiến trúc UNet. Ví dụ ảnh gốc, Ground Truth và kết quả phân vùng của UNet-base.

## Slide 3: Thử Nghiệm Loss Function: Thất Bại với Focal Loss

*   **Mục tiêu:** Cải thiện hiệu suất cho các lớp mất cân bằng dữ liệu.
*   **Cấu hình:** GeminiUNetV2 lớn hơn (`initial_filters: 64`) + `FocalLoss`.
*   **Kết quả:** mIoU giảm đáng kể xuống **0.2914**, thấp hơn nhiều so với UNet-base.
*   **Bài học:** Lựa chọn và tinh chỉnh loss function cần cẩn thận.
*   **Gợi ý hình ảnh:** Biểu đồ cột so sánh mIoU của UNet-base và GeminiUNetV2-Large-FocalLoss. Ví dụ phân vùng kém chất lượng của mô hình này.

## Slide 4: Khám phá Kiến trúc: GeminiUNetV2-Large-SAC

*   **Mô tả:** Mô hình lớn hơn với `initial_filters: 64` và tích hợp `SAC` attention.
*   **Kết quả:** Đạt mIoU **0.5602**.
*   **So sánh:** Hiệu suất thấp hơn UNet-base mặc dù lớn hơn, có thể do cấu hình hoặc quá trình huấn luyện chưa tối ưu.
*   **Gợi ý hình ảnh:** Sơ đồ khối của SAC module. Biểu đồ cột so sánh mIoU của UNet-base, GeminiUNetV2-Large-SAC.

## Slide 5: Tối ưu hóa Decoder: GeminiUNetV2-Light-Transposed-SAC

*   **Mục tiêu:** Cải thiện mIoU của mô hình nhẹ mà không giảm FPS, tìm kiếm sự cân bằng hiệu suất/hiệu quả.
*   **Giải pháp:** Thay thế `Bilinear Upsampling` bằng `Transposed Convolution` trong decoder của mô hình nhẹ (`initial_filters: 32`).
*   **Kết quả:** mIoU tăng lên **0.5336** với FPS cao (~97).
*   **So sánh & Bài học:** Cân bằng tốt hơn giữa hiệu suất và hiệu quả so với GeminiUNetV2-Large-SAC. Là nền tảng cho các thử nghiệm mô hình nhẹ tiếp theo.
*   **Gợi ý hình ảnh:** Minh họa sự khác biệt giữa Bilinear Upsampling và Transposed Convolution. Biểu đồ cột so sánh mIoU và FLOPs/Params của UNet-base, GeminiUNetV2-Large-SAC, và GeminiUNetV2-Light-Transposed-SAC.

## Slide 6: Cột Mốc Quan Trọng: Đạt mIoU 0.7 với GeminiUNetV2-Light-Transposed-SAC-DiceFocused

*   **Mục tiêu:** Đạt mIoU > 0.7 với mô hình nhẹ.
*   **Cấu hình:** Dựa trên GeminiUNetV2-Light-Transposed-SAC, loss function tập trung vào Dice, huấn luyện 140 epochs trên dữ liệu 25-band gốc.
*   **Kết quả:** Đạt **Best mIoU: 0.7026**. **Đã đạt được mục tiêu!**
*   **So sánh:** Cải thiện đáng kể mIoU so với GeminiUNetV2-Light-Transposed-SAC (0.5336). Là mô hình nhẹ hơn nhiều so với UNet-base.
*   **Gợi ý hình ảnh:** Biểu đồ đường thể hiện quá trình hội tụ của mIoU và loss qua các epoch. Biểu đồ cột so sánh mIoU của UNet-base, GeminiUNetV2-Light-Transposed-SAC, và GeminiUNetV2-Light-Transposed-SAC-DiceFocused. Ví dụ ảnh gốc, Ground Truth và kết quả phân vùng chất lượng cao.

## Slide 7: Đột Phá Tốc Độ: Tối ưu hóa với PCA (3-band)

*   **Mục tiêu:** Giảm chiều dữ liệu từ 25 xuống 3 kênh bằng PCA mà không giảm mIoU, nhằm tăng tốc độ xử lý.
*   **Kết quả:** Đạt **Best mIoU: 0.7030**, FPS: **~290.12**.
*   **So sánh & Bài học:** Giữ nguyên độ chính xác so với GeminiUNetV2-Light-Transposed-SAC-DiceFocused (0.7026) nhưng tăng tốc độ xử lý lên gần 3 lần. PCA hiệu quả trong việc loại bỏ thông tin dư thừa.
*   **Gợi ý hình ảnh:** Minh họa quá trình PCA (25 kênh -> 3 kênh). Biểu đồ cột so sánh mIoU và FPS của GeminiUNetV2-Light-Transposed-SAC-DiceFocused và mô hình PCA 3-band. Biểu đồ phân tán mIoU vs. FLOPs/Params/FPS cho tất cả các mô hình.

## Slide 8: Tối ưu hóa GPU: GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector

*   **Mục tiêu:** Tăng mức sử dụng VRAM và thông lượng bằng cách tăng `batch_size` lên 32, tích hợp `Trainable Band Selector` để mô hình tự học các kênh quan trọng.
*   **Cấu hình:** Dựa trên GeminiUNetV2-Light-Transposed-SAC (`initial_filters: 32`, `depth: 4`), `batch_size: 32`, sử dụng `BandSelector` để học cách giảm 25 kênh gốc xuống 5 kênh đầu vào cho UNet.
*   **Trạng thái:** Đang trong quá trình thử nghiệm và huấn luyện.
*   **Gợi ý hình ảnh:** Sơ đồ luồng dữ liệu: 25 kênh -> BandSelector -> 5 kênh -> UNet. Sơ đồ chi tiết của BandSelector module.

## Slide 9: Khám phá Giới hạn: GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector-Depth5

*   **Mục tiêu:** Tiếp tục tăng dung lượng mô hình và mức sử dụng VRAM bằng cách tăng `depth` lên 5, kết hợp `batch_size` lớn hơn và `Trainable Band Selector`.
*   **Cấu hình:** Dựa trên mô hình trước nhưng `depth: 5`.
*   **Trạng thái:** Đang trong quá trình thử nghiệm và huấn luyện.
*   **Gợi ý hình ảnh:** Biểu đồ cột so sánh FLOPs và Params của các mô hình GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector và GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector-Depth5.

## Slide 10: Bảng So sánh Hiệu suất và Hiệu quả của các Mô hình

*   **Tiêu đề:** Bảng So sánh Hiệu suất và Hiệu quả của các Mô hình
*   **Nội dung:**
    | Mô hình | Input Dim | FLOPs (G) | Params (M) | mIoU | Efficiency (mIoU/GFLOP) |
    | :------ | :-------- | :-------- | :--------- | :--- | :---------------------- |
    | UNet-base | 25 | 19.41 | 7.30 | 0.8232 | 0.0424 |
    | GeminiUNetV2-Large-SAC | 25 | 116.96 | 31.27 | 0.5602 | 0.0048 |
    | GeminiUNetV2-Light-Transposed-SAC | 25 | 29.94 | 7.82 | 0.5336 | 0.0178 |
    | GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector | 25 -> 5 (Trainable) | 22.85 | 7.82 | Đang huấn luyện | Đang huấn luyện |
    | GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector-Depth5 | 25 -> 5 (Trainable) | 27.84 | 31.32 | Đang huấn luyện | Đang huấn luyện |
*   **Gợi ý biểu đồ:** Biểu đồ cột so sánh mIoU của tất cả các mô hình. Biểu đồ phân tán mIoU vs. FLOPs/Params/FPS.

## Slide 11: Kết luận và Hướng đi Tiếp theo

*   **Kết luận:** Đã đạt mIoU 0.7, PCA hiệu quả, đang tối ưu GPU với Trainable Band Selector.
*   **Các bước tiếp theo:**
    *   Hoàn thành huấn luyện các mô hình mới.
    *   Đánh giá chi tiết hiệu suất.
    *   Tập trung vào PCA (5 thành phần).
    *   Cải thiện hiệu suất cho các lớp yếu.
    *   Phân tích trọng số Band Selector.
    *   So sánh Band Selector vs. PCA.
    *   Quyết định kiến trúc tối ưu cuối cùng.
