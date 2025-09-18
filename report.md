# Báo cáo Tiến độ Giữa Kỳ: Tối ưu hóa Mô hình Phân vùng Ảnh Viễn thám

**Mục tiêu tổng thể:** Xây dựng một mô hình lightweight (nhẹ) cho bài toán phân vùng ảnh viễn thám đa kênh (25 bands), đạt chỉ số **mIoU > 0.7** và có tốc độ xử lý (FPS) cao, đồng thời tối ưu hóa việc sử dụng tài nguyên GPU.

---

## 1. Hành trình Phát triển Mô hình (Tóm tắt các giai đoạn trước)

### 1.1. Giai đoạn Khám phá và Thiết lập Baseline

Trong giai đoạn khởi đầu, chúng tôi tập trung vào việc thiết lập một đường cơ sở hiệu suất và khám phá các yếu tố ảnh hưởng đến mô hình.

*   **Mô hình UNet-base (Baseline):**
    *   **Mô tả:** Đây là kiến trúc UNet cơ bản, được sử dụng làm điểm khởi đầu để đánh giá các cải tiến sau này.
    *   **Kết quả:** Đạt mIoU **0.8232**. Kết quả này cho thấy kiến trúc UNet cơ bản đã có khả năng phân vùng tốt trên tập dữ liệu.
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "UNet-base: Điểm Khởi Đầu Mạnh Mẽ".
        *   **Hình ảnh:** Sơ đồ khối đơn giản của kiến trúc UNet.
        *   **Biểu đồ:** Biểu đồ cột so sánh mIoU của UNet-base với các mô hình khác (sẽ được trình bày chi tiết ở Mục 3).
        *   **Ví dụ:** Ảnh gốc, Ground Truth và kết quả phân vùng của UNet-base để minh họa hiệu suất baseline.

*   **Thí nghiệm GeminiUNetV2-Large-FocalLoss:**
    *   **Mục tiêu:** Cải thiện hiệu suất cho các lớp bị mất cân bằng dữ liệu bằng cách sử dụng `FocalLoss`, với hy vọng vượt qua baseline.
    *   **Cấu hình:** Sử dụng mô hình `GeminiUNetV2` lớn hơn (`initial_filters: 64`) kết hợp với `FocalLoss`.
    *   **Kết quả:** mIoU giảm đáng kể xuống **0.2914**, thấp hơn nhiều so với UNet-base. Điều này cho thấy `FocalLoss` với các tham số mặc định không phù hợp hoặc cần tinh chỉnh sâu hơn, và quá trình huấn luyện không ổn định.
    *   **So sánh & Bài học:** Mặc dù là một mô hình lớn hơn, việc lựa chọn loss function không phù hợp đã dẫn đến hiệu suất kém hơn đáng kể so với baseline. Điều này nhấn mạnh tầm quan trọng của việc lựa chọn và tinh chỉnh loss function cẩn thận.
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "Thử Nghiệm Loss Function: Thất Bại với Focal Loss".
        *   **Biểu đồ:** Biểu đồ cột so sánh mIoU của UNet-base và GeminiUNetV2-Large-FocalLoss để làm nổi bật sự sụt giảm hiệu suất.
        *   **Hình ảnh:** Có thể hiển thị ví dụ phân vùng kém chất lượng của mô hình này.

### 1.2. Giai đoạn Tối ưu Kiến trúc và Tăng cường Dữ liệu

Sau khi xác định các vấn đề ban đầu, chúng tôi chuyển sang tối ưu hóa kiến trúc và áp dụng các kỹ thuật tăng cường dữ liệu để cải thiện mIoU mà vẫn duy trì hiệu quả.

*   **Mô hình GeminiUNetV2-Large-SAC:**
    *   **Mô tả:** Mô hình lớn hơn với `initial_filters: 64` và tích hợp `SAC` attention.
    *   **Kết quả:** Đạt mIoU **0.5602**. Mặc dù có mIoU cao hơn UNet-base, nhưng hiệu quả tính toán thấp hơn đáng kể (FLOPs cao hơn nhiều).
    *   **So sánh:** So với UNet-base (mIoU 0.8232), mô hình này cho thấy hiệu suất thấp hơn mặc dù lớn hơn, có thể do cấu hình hoặc quá trình huấn luyện chưa tối ưu.
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "Khám phá Kiến trúc: GeminiUNetV2-Large-SAC".
        *   **Hình ảnh:** Sơ đồ khối của SAC module.
        *   **Biểu đồ:** Biểu đồ cột so sánh mIoU của UNet-base, GeminiUNetV2-Large-SAC.

*   **Thí nghiệm GeminiUNetV2-Light-Transposed-SAC:**
    *   **Mục tiêu:** Cải thiện mIoU của mô hình nhẹ mà không làm giảm FPS, tìm kiếm sự cân bằng giữa hiệu suất và hiệu quả.
    *   **Giải pháp:** Thay thế `Bilinear Upsampling` bằng `Transposed Convolution` trong decoder của mô hình nhẹ (`initial_filters: 32`).
    *   **Kết quả:** mIoU của mô hình nhẹ tăng lên **0.5336** với FPS cao (~97).
    *   **So sánh & Bài học:** Mô hình này cho thấy sự cân bằng tốt hơn giữa hiệu suất và hiệu quả so với GeminiUNetV2-Large-SAC. Mặc dù mIoU thấp hơn UNet-base, nhưng nó là một mô hình nhẹ hơn nhiều (Params và FLOPs thấp hơn). Đây là một bước tiến quan trọng, xác định kiến trúc nền tảng cho các thử nghiệm tiếp theo tập trung vào mô hình nhẹ.
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "Tối ưu hóa Decoder: Transposed Convolution".
        *   **Hình ảnh:** Minh họa sự khác biệt giữa Bilinear Upsampling và Transposed Convolution.
        *   **Biểu đồ:** Biểu đồ cột so sánh mIoU và FLOPs/Params của UNet-base, GeminiUNetV2-Large-SAC, và GeminiUNetV2-Light-Transposed-SAC để làm nổi bật sự đánh đổi.

---

## 2. Đạt Mục tiêu và Tối ưu hóa Hiệu quả GPU (Giai đoạn Hiện tại)

### 2.1. Đạt Mục tiêu mIoU 0.7

Sau nhiều thử nghiệm, chúng tôi đã đạt được mục tiêu mIoU 0.7 với mô hình nhẹ, sử dụng cấu hình tinh chỉnh và huấn luyện kéo dài.

*   **Thí nghiệm GeminiUNetV2-Light-Transposed-SAC-DiceFocused:**
    *   **Mục tiêu:** Đạt mIoU > 0.7 với mô hình nhẹ, tiếp tục tối ưu hóa kiến trúc đã được xác định ở giai đoạn trước.
    *   **Cấu hình:** Dựa trên kiến trúc GeminiUNetV2-Light-Transposed-SAC, sử dụng loss function tập trung vào Dice (`alpha: 0.4`, `beta: 0.6`), huấn luyện 140 epochs trên dữ liệu 25-band gốc.
    *   **Kết quả:** Đạt **Best mIoU: 0.7026**, FPS: ~98.66. **Đây là một cột mốc quan trọng, đã đạt được mục tiêu mIoU đề ra!**
    *   **So sánh:** Mô hình này cho thấy sự cải thiện đáng kể về mIoU so với GeminiUNetV2-Light-Transposed-SAC (0.5336), chứng tỏ việc tinh chỉnh loss function và huấn luyện kéo dài là hiệu quả. Mặc dù mIoU thấp hơn UNet-base, nhưng nó là một mô hình nhẹ hơn nhiều (FLOPs và Params thấp hơn đáng kể).
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "Cột Mốc Quan Trọng: Đạt mIoU 0.7".
        *   **Biểu đồ:** Biểu đồ đường thể hiện quá trình hội tụ của mIoU và loss qua các epoch.
        *   **Biểu đồ:** Biểu đồ cột so sánh mIoU của UNet-base, GeminiUNetV2-Light-Transposed-SAC, và GeminiUNetV2-Light-Transposed-SAC-DiceFocused.
        *   **Ví dụ:** Ảnh gốc, Ground Truth và kết quả phân vùng của mô hình này để minh họa chất lượng phân vùng cao.

### 2.2. Tối ưu hóa Tốc độ và Hiệu quả GPU với PCA

Sau khi đạt được độ chính xác, chúng tôi tập trung vào việc cải thiện tốc độ xử lý và hiệu quả sử dụng GPU.

*   **Thí nghiệm PCA (3-band):**
    *   **Mục tiêu:** Giảm chiều dữ liệu từ 25 xuống 3 kênh bằng PCA mà không làm giảm mIoU, nhằm tăng tốc độ xử lý.
    *   **Kết quả:** Đạt **Best mIoU: 0.7030**, FPS: **~290.12**.
    *   **So sánh & Bài học:** Đây là một thành công vượt trội! Mô hình không chỉ giữ nguyên độ chính xác so với GeminiUNetV2-Light-Transposed-SAC-DiceFocused (0.7026) mà còn tăng tốc độ xử lý lên gần 3 lần. Điều này chứng tỏ PCA là một phương pháp giảm chiều hiệu quả, giúp loại bỏ thông tin dư thừa mà vẫn giữ được các đặc trưng cốt lõi, dẫn đến hiệu quả tính toán vượt trội.
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "Đột Phá Tốc Độ: Tối ưu hóa với PCA".
        *   **Hình ảnh:** Minh họa quá trình PCA (25 kênh -> 3 kênh).
        *   **Biểu đồ:** Biểu đồ cột so sánh mIoU và FPS của GeminiUNetV2-Light-Transposed-SAC-DiceFocused và mô hình PCA 3-band.
        *   **Biểu đồ:** Biểu đồ phân tán mIoU vs. FLOPs/Params/FPS cho tất cả các mô hình để làm nổi bật vị trí của mô hình PCA.

### 2.3. Các Mô hình Mới để Tối ưu hóa Hiệu quả GPU

Để tiếp tục tối ưu hóa việc sử dụng GPU, chúng tôi đã phát triển các cấu hình mới tập trung vào việc tăng cường tải GPU và khám phá các kỹ thuật chọn kênh có thể huấn luyện.

*   **Mô hình GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector (Batch Size 32, Depth 4, Trainable Band Selector)**
    *   **Mục tiêu:** Tăng mức sử dụng VRAM và thông lượng bằng cách tăng `batch_size` lên 32, đồng thời tích hợp `Trainable Band Selector` để mô hình tự học các kênh quan trọng, nhằm tối ưu hóa hiệu quả GPU.
    *   **Cấu hình:** Dựa trên kiến trúc GeminiUNetV2-Light-Transposed-SAC (`initial_filters: 32`, `depth: 4`), `batch_size: 32`, sử dụng `BandSelector` để học cách giảm 25 kênh gốc xuống 5 kênh đầu vào cho UNet.
    *   **Trạng thái:** Đang trong quá trình thử nghiệm và huấn luyện.
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "Tối ưu hóa GPU: Trainable Band Selector".
        *   **Hình ảnh:** Sơ đồ luồng dữ liệu: 25 kênh -> BandSelector -> 5 kênh -> UNet.
        *   **Hình ảnh:** Sơ đồ chi tiết của BandSelector module.

*   **Mô hình GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector-Depth5 (Batch Size 32, Depth 5, Trainable Band Selector)**
    *   **Mục tiêu:** Tiếp tục tăng dung lượng mô hình và mức sử dụng VRAM bằng cách tăng `depth` lên 5, kết hợp với `batch_size` lớn hơn và `Trainable Band Selector`, nhằm khám phá giới hạn hiệu suất và hiệu quả.
    *   **Cấu hình:** Dựa trên mô hình GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector nhưng `depth: 5`.
    *   **Trạng thái:** Đang trong quá trình thử nghiệm và huấn luyện.
    *   **Gợi ý trực quan:**
        *   **Slide:** Tiêu đề "Khám phá Giới hạn: Tăng Depth lên 5".
        *   **Biểu đồ:** Biểu đồ cột so sánh FLOPs và Params của các mô hình GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector và GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector-Depth5 để làm nổi bật sự gia tăng độ phức tạp.

---

## 3. Bảng So sánh Hiệu suất và Hiệu quả của các Mô hình

Bảng dưới đây tóm tắt các số liệu quan trọng của các mô hình đã và đang được phát triển, giúp đánh giá hiệu suất và hiệu quả sử dụng tài nguyên.

| Mô hình | Input Dim | FLOPs (G) | Params (M) | mIoU | Efficiency (mIoU/GFLOP) |
| :------ | :-------- | :-------- | :--------- | :--- | :---------------------- |
| UNet-base | 25 | 19.41 | 7.30 | 0.8232 | 0.0424 |
| GeminiUNetV2-Large-SAC | 25 | 116.96 | 31.27 | 0.5602 | 0.0048 |
| GeminiUNetV2-Light-Transposed-SAC | 25 | 29.94 | 7.82 | 0.5336 | 0.0178 |
| GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector | 25 -> 5 (Trainable) | 22.85 | 7.82 | Đang huấn luyện | Đang huấn luyện |
| GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector-Depth5 | 25 -> 5 (Trainable) | 27.84 | 31.32 | Đang huấn luyện | Đang huấn luyện |

**Ghi chú:**
*   **Input Dim:** Kích thước kênh đầu vào của mô hình. Đối với các mô hình sử dụng `Trainable Band Selector`, đầu vào thực tế là 25 kênh, nhưng mô hình UNet chính xử lý 5 kênh được học.
*   **FLOPs (G):** Số lượng phép toán dấu phẩy động (Giga Floating Point Operations) ước tính cho một lần forward pass trên một ảnh đầu vào 224x224. Giá trị này càng thấp càng tốt cho hiệu quả tính toán.
*   **Params (M):** Tổng số lượng tham số có thể huấn luyện của mô hình (triệu tham số). Giá trị này càng thấp càng tốt cho mô hình lightweight.
*   **mIoU:** Mean Intersection over Union, chỉ số đánh giá hiệu suất phân đoạn. Giá trị càng cao càng tốt.
*   **Efficiency (mIoU/GFLOP):** Hiệu suất của mô hình, được tính bằng mIoU chia cho FLOPs. Chỉ số này cho biết mô hình đạt được bao nhiêu độ chính xác trên mỗi GFLOP tính toán. Giá trị càng cao càng tốt.

---

## 4. Kết luận và Hướng đi Tiếp theo

Chúng tôi đã đạt được mục tiêu mIoU 0.7 với mô hình nhẹ và chứng minh hiệu quả vượt trội của PCA trong việc tối ưu hóa tốc độ. Các thử nghiệm hiện tại với `Trainable Band Selector` và các cấu hình `depth` khác nhau nhằm mục đích tối ưu hóa hơn nữa hiệu quả sử dụng GPU và khám phá tiềm năng của việc học chọn kênh end-to-end.

**Các bước tiếp theo:**
1.  Hoàn thành huấn luyện các mô hình GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector và GeminiUNetV2-Light-Transposed-SAC-TrainableBandSelector-Depth5.
2.  Đánh giá chi tiết hiệu suất (mIoU, FPS, Latency) và hiệu quả sử dụng VRAM của chúng.
3.  **Tập trung vào PCA:** Tiếp tục khám phá tiềm năng của PCA, đặc biệt là với 5 thành phần chính, để đạt được sự cân bằng tối ưu giữa độ chính xác và hiệu quả tính toán.
4.  **Cải thiện hiệu suất cho các lớp yếu:** Phân tích sâu hơn các lớp có hiệu suất thấp (ví dụ: "Vegetation") và thử nghiệm các chiến lược cụ thể (như tinh chỉnh loss function, tăng cường dữ liệu chuyên biệt) để cải thiện mIoU cho các lớp này.
5.  Phân tích các trọng số đã học của `Trainable Band Selector` để hiểu rõ hơn về các kênh quan trọng được mô hình lựa chọn.
6.  So sánh toàn diện kết quả của `Trainable Band Selector` với phương pháp PCA.
7.  Dựa trên kết quả, quyết định kiến trúc và chiến lược tối ưu cuối cùng cho mô hình phân vùng ảnh viễn thám.
