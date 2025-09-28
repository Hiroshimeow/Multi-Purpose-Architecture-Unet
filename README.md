python run.py --config configs/unet_stable_loss.yaml --run_name tangpatchsize
python run.py --config configs/unet_paper_replica.yaml --run_name unet_StepLR

python analyze.py training_runs/GeminiUNet_Run_1_20250912_122252
python analyze.py training_runs/abc

python analyze.py training_runs/unet_run_A_20250911_140000 training_runs/unet_run_B_20250911_150000

python analyze.py 'training_runs/CB_SFNet_25_channels_z_score_20250924_234020' 
--- 
Sử dụng A5000 để training
## 📂 Cấu trúc Project

```
Multi-Purpose-Architecture-Unet/
├── Image_dataset/              # Thư mục chứa dữ liệu
│   ├── cubes_fl32/
│   └── labels/
├── configs/
│   └── unet_base.yaml          # File cấu hình thử nghiệm
├── src/
│   ├── __init__.py
│   ├── datasets.py             # Module định nghĩa các class Dataset
│   ├── losses.py               # Module định nghĩa các hàm Loss
│   ├── models.py               # Module định nghĩa các kiến trúc Model
│   ├── experiment_manager.py   # Class quản lý file, log, checkpoint
│   ├── plotter.py              # Module vẽ biểu đồ
│   └── trainer.py              # Class chứa logic vòng lặp training
├── run.py                      # Điểm khởi đầu duy nhất để chạy training
└── requirements.txt            # Các thư viện cần thiết
```

source /data1/.venv/bin/activate

## 🔬 Phân tích chi tiết các file

Mỗi file trong project có một vai trò và trách nhiệm rõ ràng.

-   **`run.py` - Nhạc trưởng**
    -   **Vai trò:** Điểm khởi đầu (entry point) duy nhất của project.
    -   **Trách nhiệm:**
        -   Đọc và xử lý các tham số dòng lệnh (`argparse`).
        -   Tải và hợp nhất file cấu hình `.yaml`.
        -   Khởi tạo tất cả các đối tượng cần thiết: `ExperimentManager`, `Dataset`, `Model`, `Optimizer`, `Loss`, `Scheduler`.
        -   Khởi tạo `Trainer` và ra lệnh cho nó bắt đầu quá trình training.

-   **`configs/unet_base.yaml` - Bản thiết kế**
    -   **Vai trò:** Định nghĩa toàn bộ một thử nghiệm.
    -   **Trách nhiệm:**
        -   Chứa tất cả các siêu tham số, đường dẫn, và tên các thành phần (model, loss, optimizer).
        -   Giúp tách biệt hoàn toàn cấu hình ra khỏi mã nguồn.

-   **`src/trainer.py` - Động cơ**
    -   **Vai trò:** Đóng gói toàn bộ logic của một vòng lặp training.
    -   **Trách nhiệm:**
        -   Thực hiện vòng lặp `for epoch in ...`.
        -   Gọi các hàm train và evaluate cho mỗi epoch.
        -   Tương tác với `ExperimentManager` để ghi log và lưu checkpoint.
        -   Xử lý logic early stopping.
        -   Tải lại trạng thái từ checkpoint khi resume.

-   **`src/experiment_manager.py` - Người quản lý**
    -   **Vai trò:** Xử lý tất cả các thao tác liên quan đến file và trạng thái của thử nghiệm.
    -   **Trách nhiệm:**
        -   Tạo và quản lý các thư mục kết quả duy nhất.
        -   Lưu file cấu hình cuối cùng.
        -   Ghi log các chỉ số ra file `.csv`.
        -   Lưu và tải các checkpoint (model, optimizer, scheduler).
        -   Tạo báo cáo cuối cùng sau khi training kết thúc.
        -   **Quan trọng:** Class này hoàn toàn không biết gì về `UNet` hay logic training.

-   **`src/models.py` - Nhà máy Model**
    -   **Vai trò:** Chứa định nghĩa của tất cả các kiến trúc model.
    -   **Trách nhiệm:**
        -   Định nghĩa các class model (ví dụ: `UNet`).
        -   Cung cấp một hàm `get_model(name, params)` (factory) để tạo một instance của model dựa trên tên được cung cấp trong file config.

-   **`src/datasets.py` & `src/losses.py` - Thư viện thành phần**
    -   **Vai trò:** Chứa các class `Dataset` và `Loss` có thể tái sử dụng.
    -   **Trách nhiệm:** Cung cấp các khối xây dựng cơ bản cho quá trình training.

-   **`src/plotter.py` - Họa sĩ**
    -   **Vai trò:** Một module độc lập chỉ chứa các hàm vẽ biểu đồ.
    -   **Trách nhiệm:**
        -   Tạo biểu đồ lịch sử training (loss, accuracy, mIoU).
        -   Vẽ ma trận nhầm lẫn.
        -   Vẽ biểu đồ so sánh các chỉ số cho từng lớp.


## Hướng Dẫn Sử Dụng Framework Training

Mọi tương tác với framework đều thông qua script `run.py` và các file cấu hình trong thư mục `configs/`. Dưới đây là các kịch bản sử dụng phổ biến nhất.

### 1. Chạy một thử nghiệm cơ bản

**Giải thích:**
Đây là lệnh cơ bản nhất. Nó sẽ sử dụng file cấu hình mặc định (`configs/unet_base.yaml`) và đặt tên cho lần chạy là `unet_baseline`. Framework sẽ tự động tạo một thư mục kết quả duy nhất có chứa tên này và một dấu thời gian để tránh ghi đè.

**Lệnh:**
```bash
python run.py --run_name unet_baseline
```

**Kết quả:**
Một thư mục mới sẽ được tạo, ví dụ: `training_runs/unet_baseline_20250910_140000/`. Bên trong chứa toàn bộ kết quả, log, model và biểu đồ của lần chạy này.

---

### 2. Chạy thử nghiệm với một file cấu hình riêng

**Giải thích:**
Giả sử bạn muốn thử nghiệm một phiên bản U-Net "mỏng" hơn (ít filter hơn) để giảm độ nặng của model. Bạn tạo một file config mới `configs/unet_slim.yaml` và chỉ định nó khi chạy.

**File `configs/unet_slim.yaml` (Ví dụ):**
```yaml
# Kế thừa từ file base, chỉ thay đổi phần model
# (Lưu ý: Framework hiện tại chưa hỗ trợ kế thừa, bạn cần copy toàn bộ nội dung)
# ... (copy toàn bộ nội dung từ unet_base.yaml) ...

model:
  name: "UNetSlim" # Giả sử bạn đã định nghĩa UNetSlim trong src/models.py
  params:
    in_channels: 25
    num_classes: 5
    base_filters: 32 # Giảm số filter
```

**Lệnh:**
```bash
python run.py --config configs/unet_slim.yaml --run_name unet_slim_32filters
```

**Kết quả:**
Một lần chạy mới được tạo (`training_runs/unet_slim_32filters_.../`) sử dụng hoàn toàn các tham số từ file `unet_slim.yaml`.

---

### 3. Ghi đè nhanh siêu tham số để thử nghiệm

**Giải thích:**
Đây là cách cực kỳ hữu ích để thực hiện "hyperparameter tuning". Bạn không cần tạo file config mới cho mỗi lần thay đổi nhỏ. Lệnh này sẽ sử dụng cấu hình cơ bản nhưng ghi đè learning rate thành `0.001` và batch size thành `8`. Tên của lần chạy (`--run_name`) nên mô tả sự thay đổi này để dễ dàng nhận biết.

**Lệnh:**
```bash
python run.py --run_name unet_lr_1e-3_bs_8 --lr 0.001 --batch_size 8
```

**Kết quả:**
Framework sẽ tạo thư mục `training_runs/unet_lr_1e-3_bs_8_.../`. Bên trong thư mục này, file `config.yaml` được lưu lại sẽ chứa giá trị `lr: 0.001` và `batch_size: 8`, đảm bảo bạn biết chính xác các tham số đã được sử dụng.

---

### 4. Tiếp tục (Resume) một lần training bị gián đoạn

**Giải thích:**
Giả sử lần chạy `unet_lr_1e-3_bs_8_20250910_141500` của bạn bị dừng ở epoch 50 do mất điện. Bạn có thể tiếp tục training chính xác từ nơi nó đã dừng lại bằng cách chỉ định đường dẫn đến thư mục của lần chạy đó. Framework sẽ tự động tải lại model, optimizer, scheduler và bắt đầu từ epoch 51.

**Lệnh:**```bash
python run.py --resume_path training_runs/unet_lr_1e-3_bs_8_20250910_141500
```

**Kết quả:**
Không có thư mục mới nào được tạo. Quá trình training sẽ tiếp tục trong thư mục đã có. Các file log (`history.csv`) sẽ được ghi tiếp, và các checkpoint sẽ được cập nhật.

---

### 5. Phân tích lại kết quả của một lần chạy đã hoàn thành

**Giải thích:**
Sau khi training xong, bạn muốn tạo lại các biểu đồ với định dạng khác hoặc chỉ đơn giản là xem lại kết quả mà không cần chạy lại model. Bạn có thể chạy trực tiếp script `experiment_manager.py` (trong tương lai) hoặc một script phân tích riêng.

*(Lưu ý: Để giữ cho `run.py` chỉ làm nhiệm vụ training, chúng ta sẽ tạo một script phân tích riêng là `analyze.py`)*

**File `analyze.py` (Ví dụ):**
```python
# analyze.py
import argparse
from src.experiment_manager import ExperimentManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze a completed training run.")
    parser.add_argument('--run_path', type=str, required=True, help='Path to the training run directory to analyze.')
    args = parser.parse_args()
    
    # Giả sử ExperimentManager có một phương thức static để phân tích
    ExperimentManager.analyze_run(args.run_path) ```

**Lệnh:**
```bash
python analyze.py --run_path training_runs/unet_baseline_20250910_140000
```

**Kết quả:**
Script sẽ đọc các file `config.yaml`, `history.csv`, `final_metrics.json` từ thư mục được chỉ định, in ra tóm tắt kết quả và tạo lại các biểu đồ với một hậu tố mới (ví dụ: `_re-plot.png`) để không ghi đè lên các file gốc.

---

### 6. Chạy thử nghiệm với số kênh đầu vào khác

**Giải thích:**
Bạn muốn thử nghiệm model với 15 kênh thay vì 25. Bạn chỉ cần tạo một file config mới và thay đổi các tham số liên quan.

**File `configs/unet_15channels.yaml` (Ví dụ):**
```yaml
# ... (copy toàn bộ nội dung từ unet_base.yaml) ...

model:
  name: "UNet"
  params:
    in_channels: 15 # THAY ĐỔI
    num_classes: 5
    bilinear: True

# ...
```

**Lệnh:**
```bash
python run.py --config configs/unet_15channels.yaml --run_name unet_15channels
```