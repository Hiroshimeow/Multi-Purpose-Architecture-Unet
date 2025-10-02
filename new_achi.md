# Báo cáo Phân tích và Đề xuất Kiến trúc cho Bài toán Phân đoạn Ảnh Siêu phổ trong Lĩnh vực Xe tự hành

Tài liệu này cung cấp một phân tích sâu rộng về các kiến trúc mạng, cơ chế attention, và các phương pháp tiếp cận hiện đại cho bài toán phân đoạn ngữ nghĩa (semantic segmentation) sử dụng dữ liệu ảnh siêu phổ (Hyperspectral Imaging - HSI), đặc biệt tập trung vào bộ dữ liệu HSI-Drive cho xe tự hành.

---

## Phần 1: Khảo sát các Cơ chế Attention

Attention đã trở thành một thành phần không thể thiếu trong các mô hình học sâu cho thị giác máy tính, giúp mô hình tập trung vào các đặc trưng quan trọng.

### 1.1. Phân tích CBAM (Convolutional Block Attention Module)

CBAM là một cơ chế attention nhẹ và hiệu quả, bao gồm hai mô-đun con áp dụng tuần tự:

-   **Channel Attention Module (CAM):** Xác định "cái gì" quan trọng bằng cách tính trọng số cho mỗi kênh.
    -   **Công thức:** `Mc(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )`
-   **Spatial Attention Module (SAM):** Xác định "ở đâu" quan trọng bằng cách tạo ra một bản đồ trọng số không gian.
    -   **Công thức:** `Ms(F') = σ( f^7x7( [AvgPool(F'); MaxPool(F')] ) )`

-   **Hạn chế:** Việc xử lý tuần tự và cách nén thông tin của SAM có thể làm mất mát thông tin vị trí chính xác và các mối quan hệ không gian tầm xa.

### 1.2. Khám phá các Kiến trúc Attention Thay thế

#### 1.2.1. Coordinate Attention (CA)

-   **Ý tưởng:** Mã hóa thông tin vị trí và các mối quan hệ tầm xa bằng cách phân tách channel attention thành hai quá trình mã hóa 1D độc lập theo chiều ngang và chiều dọc.
-   **Kiến trúc:**
    1.  **Coordinate Information Embedding:** Sử dụng hai kernel pooling `(H, 1)` và `(1, W)` để tổng hợp đặc trưng dọc theo hai hướng không gian.
    2.  **Coordinate Attention Generation:** Ghép hai tensor đặc trưng, đưa qua một lớp Conv1x1, sau đó tách ra và dùng hai lớp Conv1x1 khác để tạo ra hai bản đồ attention cho mỗi hướng.
-   **Công thức (Output cuối cùng):** `y_c(i, j) = x_c(i, j) * g_c^h(i) * g_c^w(j)`
-   **Ưu điểm:** Hiệu quả, nắm bắt được thông tin vị trí chính xác và quan hệ tầm xa, rất phù hợp cho segmentation.

#### 1.2.2. Triplet Attention

-   **Ý tưởng:** Nắm bắt các tương tác chéo giữa các chiều (cross-dimension interaction) bằng cách sử dụng một cấu trúc ba nhánh. Mỗi nhánh tập trung vào một cặp hai trong ba chiều (C, H, W).
-   **Kiến trúc:**
    1.  **Ba nhánh xử lý song song:**
        -   Nhánh 1: Tương tác giữa (H, W).
        -   Nhánh 2: Tương tác giữa (C, H).
        -   Nhánh 3: Tương tác giữa (C, W).
    2.  **Z-Pool:** Trong mỗi nhánh, chiều không được tương tác sẽ được nén lại bằng cách ghép kết quả của `AvgPool` và `MaxPool`.
    3.  **Attention Map:** Tensor đã nén được đưa qua một lớp Conv và hàm Sigmoid để tạo bản đồ attention.
    4.  **Tổng hợp:** Kết quả của ba nhánh được lấy trung bình cộng.
-   **Công thức (Tổng hợp cuối cùng):** `Y = (Output_spatial + Output_ch + Output_cv) / 3`
-   **Ưu điểm:** Nắm bắt được các mối quan hệ phức tạp hơn so với việc xử lý kênh và không gian một cách riêng biệt hoặc tuần tự.

#### 1.2.3. SimAM (Simple, Parameter-Free Attention Module)

-   **Ý tưởng:** Xây dựng một cơ chế attention không cần tham số (parameter-free). Tầm quan trọng của mỗi neuron được xác định bằng cách định nghĩa một hàm năng lượng dựa trên các lý thuyết khoa học thần kinh.
-   **Kiến trúc:**
    1.  Định nghĩa một hàm năng lượng cho mỗi neuron để đo lường sự khác biệt tuyến tính của nó so với các neuron khác trong cùng một kênh.
    2.  Tìm một nghiệm dạng đóng (closed-form solution) cho hàm năng lượng, giúp tính toán nhanh chóng.
    3.  Các neuron có năng lượng thấp hơn được coi là quan trọng hơn.
-   **Công thức (Năng lượng tối thiểu):**
    ```
    E*(t) = (4 * (σ_hat^2 + λ)) / ((t - μ_hat)^2 + 2*σ_hat^2 + 2*λ)
    ```
    Trong đó `μ_hat` và `σ_hat^2` là trung bình và phương sai của tất cả các neuron trong kênh. Trọng số attention được tính bằng `1 / E*(t)`.
-   **Ưu điểm:** Cực kỳ nhẹ, không làm tăng độ phức tạp của mô hình, tính toán đồng thời cả attention kênh và không gian.

---

## Phần 2: Bối cảnh Ứng dụng và Phân tích Dataset

### 2.1. Dataset HSI-Drive

-   **Mục đích:** Đây là bộ dữ liệu công khai được xây dựng để thúc đẩy nghiên cứu về ứng dụng của ảnh siêu phổ (HSI) cho các Hệ thống Hỗ trợ Lái xe Nâng cao (ADAS) và Xe tự hành (ADS).
-   **Đặc điểm:**
    -   **Số kênh:** 25 kênh trong dải phổ Nhìn thấy và Cận hồng ngoại (Visible-NearInfraRed - VNIR).
    -   **Đa dạng:** Dữ liệu được thu thập trong nhiều điều kiện thực tế: 4 mùa, các thời điểm khác nhau trong ngày, thời tiết đa dạng (nắng, mây, mưa) và các loại đường khác nhau (đô thị, nông thôn, cao tốc).
-   **Phiên bản:**
    -   **v1.x:** Tập trung vào việc gán nhãn theo vật liệu bề mặt.
    -   **v2.0 (mới nhất):** Mở rộng lên 752 ảnh được gán nhãn thủ công, bổ sung các lớp quan trọng cho an toàn giao thông như `phương tiện`, `biển báo`, `người đi bộ`, `người đi xe đạp`.

### 2.2. Hướng tiếp cận và Thách thức

-   **Hướng tiếp cận phổ biến:**
    -   **Giảm chiều dữ liệu:** Do số lượng kênh lớn (25), các phương pháp giảm chiều như **PCA** (đang được sử dụng trong dự án này) là rất phổ biến và hiệu quả để giảm chi phí tính toán.
    -   **Sử dụng các kiến trúc CNN:** Các kiến trúc nền tảng U-Net, DeepLabv3+, HRNet thường được sử dụng làm baseline.
-   **Thách thức:**
    -   **"Nguyền rủa của số chiều" (Curse of Dimensionality):** Số lượng kênh lớn đòi hỏi chi phí tính toán và bộ nhớ cao.
    -   **Mất cân bằng lớp (Class Imbalance):** Một số lớp (ví dụ: người đi bộ) xuất hiện ít hơn nhiều so với các lớp khác (ví dụ: đường).
    -   **Kích thước Dataset:** Các bộ dữ liệu HSI cho xe tự hành vẫn còn tương đối nhỏ so với các bộ dữ liệu RGB, gây khó khăn cho việc huấn luyện các mô hình sâu và phức tạp.

---

## Phần 3: Tổng hợp các Công trình Nghiên cứu Liên quan (SOTA)

Một trong những bài báo tổng quan và đánh giá quan trọng nhất gần đây là:

-   **"Hyperspectral Imaging-Based Perception in Autonomous Driving Scenarios: Benchmarking Baseline Semantic Segmentation Models"** của Imad Ali Shah và cộng sự (2024/2025).

-   **Nội dung chính:**
    -   Bài báo đánh giá một loạt các mô hình segmentation baseline trên nhiều bộ dữ liệu HSI, bao gồm cả **HSI-Drive v2**.
    -   Các mô hình được đánh giá bao gồm: **DeepLabv3+, HRNet, PSPNet, và U-Net** cùng các biến thể có attention là **UNet-CA** (Coordinate Attention) và **UNet-CBAM**.
-   **Kết quả nổi bật:**
    -   Nghiên cứu chỉ ra rằng **UNet-CBAM** thường cho kết quả vượt trội hơn các mô hình khác. Điều này cho thấy việc kết hợp U-Net với một cơ chế attention có khả năng khai thác thông tin giữa các kênh (như CBAM) là một hướng đi rất hiệu quả cho dữ liệu siêu phổ.
-   **Kết luận:** Phát hiện này **xác thực mạnh mẽ hướng đi của dự án hiện tại**, vốn đang tập trung vào việc cải tiến kiến trúc U-Net với các khối attention. Nó cũng cho thấy CBAM, mặc dù đơn giản, nhưng lại rất phù hợp với đặc thù của dữ liệu HSI.

---

## Phần 4: Phân tích Kiến trúc Hiện tại của Dự án

Phần này mô tả chi tiết kiến trúc và cấu hình của mô hình đạt hiệu suất cao nhất trong dự án tính đến hiện tại (Thử nghiệm PCA 3-band), đạt **mIoU ~0.7030** và **FPS ~290.12**.

-   **Tổng quan:** Mô hình là một biến thể của U-Net, thành công nhờ 3 yếu tố:
    1.  **Giảm chiều bằng PCA:** Nén 25 kênh xuống còn 3.
    2.  **Decoder với Tích chập chuyển vị:** Dùng `ConvTranspose2d` (`bilinear=False`) để khôi phục chi tiết tốt hơn.
    3.  **Loss tập trung vào Dice:** Ưu tiên Dice Score (`beta: 0.6`) để cải thiện IoU.
-   **Kiến trúc Mô hình: `UNetBase`**
    -   **Luồng kiến trúc (`initial_filters=32`):**
        1.  **Input:** `(N, 3, H, W)`
        2.  **Encoder:** 4 tầng `Down` block, tăng số kênh từ 32 -> 64 -> 128 -> 256.
        3.  **Bottleneck:** `Down(256, 512)`.
        4.  **Decoder:** 4 tầng `Up` block, giảm số kênh và kết hợp với skip-connection từ encoder.
        5.  **Output:** `OutConv(32, 5)` -> `logits`.
-   **Cấu hình Huấn luyện:**
    -   **Loss:** `CombinedLoss` (alpha=0.4, beta=0.6).
    -   **Optimizer:** `AdamW` (lr=0.001).
    -   **Scheduler:** `CosineAnnealingLR`.
    -   **Epochs:** 140 (với early stopping).
    -   **Batch Size:** 16.
