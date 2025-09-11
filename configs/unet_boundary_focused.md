Chắc chắn rồi. Việc hiểu rõ từng tham số trong file cấu hình là chìa khóa để bạn có thể làm chủ hoàn toàn quá trình thử nghiệm. Đây là một phân tích chi tiết, sâu sắc về từng tham số trong file `configs/unet_boundary_focused.yaml`, giải thích **"Nó là gì?"**, **"Tại sao nó quan trọng (đặc biệt là theo paper)?"**, và **"Làm thế nào để tinh chỉnh nó?"**.

---

### Phân tích chi tiết `configs/unet_boundary_focused.yaml`

File cấu hình này là "bộ não" điều khiển toàn bộ thử nghiệm của bạn. Nó được thiết kế để triển khai các chiến lược cốt lõi từ bài báo U-Net.

#### **1. `Run & Environment` - Quản lý Thử nghiệm**

```yaml
run_name: "unet_boundary_focused_v1"
base_output_dir: "training_runs"
class_names: ["Road (tarmac)", "Road marks", "Painted metal", "Sky", "Concrete/stone/brick"]
```

*   **`run_name`**:
    *   **Nó là gì?**: Tên cơ sở cho thư mục kết quả. Kết quả thực tế sẽ được lưu vào `training_runs/unet_boundary_focused_v1_YYYYMMDD_HHMMSS/`.
    *   **Tại sao?**: Giúp bạn đặt tên có ý nghĩa cho mỗi thử nghiệm (ví dụ: `unet_light_w0_15`, `unet_patch_256`). Đây là nền tảng của việc tổ chức thử nghiệm.
    *   **Tinh chỉnh**: Thay đổi nó cho mỗi lần thử nghiệm để dễ dàng phân biệt.

*   **`base_output_dir`**:
    *   **Nó là gì?**: Thư mục gốc chứa tất cả các lần chạy thử nghiệm.
    *   **Tại sao?**: Giữ cho dự án của bạn gọn gàng.

*   **`class_names`**:
    *   **Nó là gì?**: Danh sách tên các lớp theo đúng thứ tự (từ 0 đến N-1).
    *   **Tại sao?**: Cực kỳ quan trọng để các biểu đồ (confusion matrix, per-class metrics) và báo cáo có nhãn đúng, giúp bạn đọc kết quả một cách chính xác.

---

#### **2. `Model & Architecture` - Định nghĩa Mô hình**

```yaml
model:
  name: "UNet"
  params:
    in_channels: 25
    num_classes: 5
    bilinear: True
```

*   **`model.name`**:
    *   **Nó là gì?**: Tên của kiến trúc model mà bạn muốn sử dụng.
    *   **Tại sao?**: Đây là cốt lõi của kiến trúc "factory". Script `run.py` sẽ đọc tên này và yêu cầu `src/models.py` tạo ra một đối tượng model tương ứng. Nó giúp hệ thống hoàn toàn độc lập với model (model-agnostic).
    *   **Tinh chỉnh**: Nếu sau này bạn thêm `AttentionUNet` vào `src/models.py`, bạn chỉ cần đổi tên ở đây thành `"AttentionUNet"`.

*   **`model.params`**:
    *   **Nó là gì?**: Một dictionary chứa các tham số sẽ được truyền vào hàm `__init__` của model (ví dụ: `UNet(in_channels=25, num_classes=5, ...)`).
    *   **`in_channels`**: Số kênh/band đầu vào của ảnh hyperspectral.
    *   **`num_classes`**: Số lớp đầu ra. Phải khớp với độ dài của `class_names`.
    *   **`bilinear`**: Lựa chọn phương pháp upsampling. `True` (Bilinear) nhanh hơn, ít tham số hơn. `False` (ConvTranspose2d) có thể học được, mạnh mẽ hơn nhưng nặng hơn.
    *   **Tham số tiềm năng**: Bạn có thể thêm `base_filters: 32` vào đây để tạo ra một phiên bản U-Net "mỏng" hơn, nhẹ hơn.

---

#### **3. `Data & Patching Strategy` - Chiến lược Dữ liệu theo Paper**

```yaml
data:
  dir: "Image_dataset"
  patching:
    patch_size: 512
    stride: 256
  class_mapping: { ... }
```

*   **`patching`**: Toàn bộ khối này là để triển khai **Overlap-Tile Strategy**.
*   **`patch_size`**:
    *   **Nó là gì?**: Kích thước (cao, rộng) của mỗi "gạch" (patch) được cắt ra từ ảnh gốc và đưa vào model.
    *   **Tại sao (theo Paper)?**: U-Net được thiết kế để hoạt động trên các vùng ảnh cục bộ. Paper sử dụng các patch lớn để đảm bảo mỗi pixel ở trung tâm có đủ "ngữ cảnh" xung quanh.
    *   **Tinh chỉnh**: Đây là một sự đánh đổi quan trọng.
        *   **Tăng `patch_size`** (ví dụ: 768): Model "nhìn" được một vùng rộng hơn, có thể tốt hơn cho các vật thể lớn, nhưng sẽ tiêu tốn **rất nhiều VRAM** và làm giảm `batch_size`.
        *   **Giảm `patch_size`** (ví dụ: 256): Tiết kiệm VRAM, cho phép `batch_size` lớn hơn, training nhanh hơn, nhưng model có thể mất ngữ cảnh của các vật thể lớn.

*   **`stride`**:
    *   **Nó là gì?**: Bước nhảy khi cắt các patch.
    *   **Tại sao (theo Paper)?**: Đây là tham số **tạo ra sự chồng lấp**. Khi `stride < patch_size`, các patch sẽ gối lên nhau. Ví dụ, `patch_size: 512` và `stride: 256` tạo ra sự chồng lấp 50% ở cả chiều ngang và dọc.
    *   **Tinh chỉnh**:
        *   **Giảm `stride`** (ví dụ: 128): Tăng độ chồng lấp. Điều này tạo ra **nhiều patch hơn**, làm cho quá trình training **dài hơn đáng kể**, nhưng có thể giúp kết quả inference cuối cùng mượt mà hơn vì mỗi pixel được dự đoán nhiều lần.
        *   **Tăng `stride`** (ví dụ: 512): Không có sự chồng lấp. Số lượng patch ít nhất, training nhanh nhất, nhưng có thể tạo ra các "hiệu ứng cạnh" (edge artifacts) trên ảnh dự đoán cuối cùng.

---

#### **4. `Augmentation` - Tăng cường Dữ liệu theo Paper**

```yaml
augmentation:
  elastic_transform:
    p: 0.5
    alpha: 120
    sigma: 6
    alpha_affine: 25
```

*   **`elastic_transform`**: Đây là kỹ thuật **"biến dạng đàn hồi"**, được paper nhấn mạnh là "khái niệm chủ chốt" để huấn luyện hiệu quả với ít dữ liệu.
*   **`p`**: Xác suất áp dụng phép biến đổi này cho một patch. `0.5` nghĩa là khoảng một nửa số patch trong mỗi epoch sẽ bị làm biến dạng.
*   **`alpha`**: Cường độ biến dạng. Giá trị càng lớn, ảnh càng bị "méo" nhiều.
*   **`sigma`**: Độ mượt của biến dạng. Giá trị càng lớn, các biến dạng càng mượt và có quy mô lớn. Giá trị nhỏ tạo ra các biến dạng nhỏ, lắt nhắt. Paper đề xuất `sigma` khoảng `alpha / 20`.
*   **`alpha_affine`**: Cường độ của các biến đổi affine (xoay, co giãn, trượt) đi kèm với biến dạng đàn hồi.

---

#### **5. `Loss Function` - Hàm Loss tập trung vào Ranh giới**

```yaml
loss:
  name: 'WeightedCombinedLoss'
  boundary_weight:
    w0: 10.0
    sigma: 5.0
  params:
    alpha: 1.0
    beta: 0.5
```

*   **`loss.name`**: Tên của hàm loss để factory gọi.
*   **`boundary_weight`**: Khối này điều khiển việc tạo ra **Pixel-wise Weight Map**.
*   **`w0`**:
    *   **Nó là gì?**: Hệ số nhân cho trọng số ở vùng ranh giới.
    *   **Tại sao (theo Paper)?**: Đây là "cây gậy" để "phạt" model. Với `w0: 10.0`, một lỗi ở ranh giới sẽ bị phạt nặng hơn khoảng 10 lần so với một lỗi ở giữa vật thể.
    *   **Tinh chỉnh**: Đây là tham số cực kỳ quan trọng.
        *   **Tăng `w0`** (ví dụ: 15, 20): Buộc model phải tập trung hơn nữa vào ranh giới. Có thể hữu ích nếu ranh giới vẫn bị mờ, nhưng nếu quá cao có thể khiến model bỏ qua việc học các vùng bên trong.
        *   **Giảm `w0`** (ví dụ: 5): Giảm sự tập trung vào ranh giới.

*   **`sigma`**:
    *   **Nó là gì?**: Độ "lan tỏa" của vùng trọng số cao xung quanh ranh giới.
    *   **Tại sao (theo Paper)?**: Nó xác định một vùng "đệm" xung quanh đường biên.
    *   **Tinh chỉnh**:
        *   **Giảm `sigma`** (ví dụ: 3): Vùng trọng số cao sẽ hẹp và sắc nét hơn, chỉ tập trung vào các pixel ngay sát ranh giới.
        *   **Tăng `sigma`** (ví dụ: 8): Vùng trọng số cao sẽ lan rộng ra xa hơn.

*   **`params.alpha`, `params.beta`**:
    *   **Nó là gì?**: Trọng số để cân bằng giữa hai thành phần của loss: `alpha` cho `WeightedCrossEntropy` và `beta` cho `DiceLoss`.
    *   **Tại sao?**: `CrossEntropy` tập trung vào độ chính xác của từng pixel (tốt cho ranh giới). `DiceLoss` tập trung vào sự chồng lấp của các vùng (tốt cho sự cân bằng tổng thể của vật thể). Việc kết hợp chúng giúp mô hình học tốt cả hai.
    *   **Tinh chỉnh**: Nếu mIoU tổng thể tốt nhưng ranh giới vẫn kém, hãy thử tăng `alpha` và giảm `beta`. Nếu các vật thể bị "vỡ" hoặc không liền mạch, hãy thử tăng `beta`.