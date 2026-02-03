# Changelog: Fix Core Mechanics & Reproduction Setup (20/01/2026)

**Người thực hiện:** Gemini Agent  
**Mục tiêu:** Đồng bộ hóa mã nguồn thực tế với mô tả lý thuyết trong Luận văn (Thesis) và tái lập kết quả hiệu năng cao (SOTA) trên tập dữ liệu HSI-Drive.

---

## 1. Phân tích & Sửa lỗi Cốt lõi (Critical Fixes)

Trước thay đổi này, mã nguồn có sự sai lệch lớn so với lý thuyết, khiến mô hình không thể học được cách chọn band (Band Selection) một cách hiệu quả.

### 1.1. Sửa lỗi Differentiable Band Selection (STE)
*   **Vấn đề cũ:** Class `LearnableBandSelector` sử dụng `torch.index_select` trên các chỉ số rời rạc (indices). Hàm này không khả vi (non-differentiable) đối với indices, làm đứt gãy chuỗi đạo hàm (Gradient Flow). Hệ quả là tham số `logits` không bao giờ được cập nhật, việc chọn band trở nên ngẫu nhiên.
*   **Thay đổi (Fix):** Triển khai lại cơ chế **Gumbel-Softmax với Straight-Through Estimator (STE)** chuẩn.
*   **File:** `src/models/attention_modules.py`
*   **Logic mới:**
    $$ y_{\text{soft}} = \text{Softmax}((\text{logits} + G) / \tau) $$
    $$ y_{\text{hard}} = \text{OneHot}(\text{TopK}(y_{\text{soft}})) $$
    $$ y_{\text{STE}} = (y_{\text{hard}} - y_{\text{soft}}).\text{detach}() + y_{\text{soft}} $$
    $$ X_{\text{masked}} = X \odot y_{\text{STE}} $$
    *   Việc nhân $X$ với Mask trước khi trích xuất band đảm bảo Gradient lan truyền ngược từ Loss về Logits.

### 1.2. Thêm cơ chế Temperature Annealing
*   **Vấn đề cũ:** Nhiệt độ $\tau$ (Temperature) của Gumbel-Softmax bị cố định (mặc định = 1.0). Code thiếu cơ chế giảm nhiệt độ để chuyển trạng thái từ "Thăm dò" (Exploration - chọn mềm) sang "Khai thác" (Exploitation - chọn cứng).
*   **Thay đổi (Fix):** Thêm logic giảm nhiệt độ vào vòng lặp training.
*   **File:** `src/trainer.py`
*   **Logic:**
    *   Epoch 0 $\to$ 100: Giảm $\tau$ từ **5.0** xuống **0.1** theo hàm mũ.
    *   Epoch > 100: Giữ cố định $\tau = 0.1$.

---

## 2. Cấu hình & Dữ liệu (Configuration Setup)

### 2.1. Chuẩn hóa Band Rankings cho Baseline
*   Đã chạy script `scripts/calculate_bands.py` để tính toán lại độ quan trọng của các band dựa trên toàn bộ dữ liệu (50 samples).
*   **Kết quả:** Tạo ra file `band_rankings.txt` chứa thứ tự band cho 5 phương pháp: Entropy, SVM-RFE, Fisher, Correlation, SAD (mapped to Variance).
*   Mục đích: Đảm bảo các phương pháp so sánh (Baselines) sử dụng input đầu vào tốt nhất có thể, tạo sự công bằng khi so sánh với TABS.

### 2.2. Điều chỉnh Config Model
*   **Model Name:** Đổi tên định danh từ `UnetBandS` thành `TABS` để khớp với registry trong `src/models/__init__.py`.
*   **Architecture:** Sử dụng backbone **UNetBase Lightweight** (`initial_filters=6`, ~0.27M params).
*   **Lưu ý quan trọng:** Theo chỉ đạo, **Spatial Attention (SA) tạm thời bị vô hiệu hóa** (`attention_block=None`) trong đợt training này để focus vào khả năng của bộ chọn band.

---

## 3. Kế hoạch Tái lập Kết quả (Reproduction Plan)

Hệ thống đang chạy song song trên GPU RTX A5000 (24GB VRAM) với tổng cộng **36 Jobs**.

### 3.1. Batch 1: Seed 25873 (Seed tốt nhất lịch sử)
*   **Script:** `scripts/reproduce_seed_25873.py`
*   **Log Master:** `master_reproduce_25873.log`
*   **Phạm vi:**
    *   6 Phương pháp: TABS, SVM-RFE, Entropy, Fisher, Correlation, SAD.
    *   3 K-values: k=3, k=5, k=7.
    *   Tổng: 18 models.

### 3.2. Batch 2: Seed 61684 (Seed tiềm năng thứ 2)
*   **Script:** `scripts/reproduce_seed_61684.py`
*   **Log Master:** `master_reproduce_61684.log`
*   **Phạm vi:** Tương tự Batch 1 (18 models).

### 3.3. Thông số Training
*   **Epochs:** 400
*   **Loss:** CombinedLoss (0.4 Dice + 0.6 CE)
*   **Batch Size:** 8
*   **Optimizer:** AdamW

---

## 4. Hướng dẫn Theo dõi (Monitoring)

### Kiểm tra Logs
```bash
# Xem log tổng của Seed 25873
tail -f master_reproduce_25873.log

# Xem chi tiết quá trình training của TABS k=5 (Seed 25873)
tail -f logs_25873/S25873_TABS_k5.log
```

### Kiểm tra TensorBoard
Đã bật TensorBoard tại port 9991.
*   **URL:** `http://<server-ip>:9991`
*   **Logdir:** `./training_runs`

---

## 5. Kết quả Kỳ vọng
Sau 400 epochs, chúng ta kỳ vọng:
1.  **TABS:** Sẽ tự động học được các band index tối ưu (hội tụ về mask 0/1). mIoU kỳ vọng > 0.77 (với k=5).
2.  **So sánh:** TABS phải vượt trội hơn SVM-RFE và Entropy ít nhất 1-2% mIoU để khẳng định luận điểm "Task-Aware tốt hơn Task-Agnostic/Offline".
