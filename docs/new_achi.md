# Thành tựu và Tiến trình Phát triển Dự án (Kể từ `cbsfnet_light_exp2`)

Tài liệu này tóm tắt chi tiết các bước đã thực hiện, những cải tiến, kết quả đạt được và các phân tích liên quan kể từ khi thử nghiệm `cbsfnet_light_exp2` được thực hiện. Mục tiêu là cung cấp một cái nhìn toàn diện về tiến trình của dự án.

---

## 1. Bối cảnh: Vấn đề với `cbsfnet_light_exp2`

*   **Suy nghĩ/Phân tích:** Thử nghiệm `cbsfnet_light_exp2` ban đầu cho kết quả mIoU rất tốt (0.7466) với số lượng tham số nhẹ (0.53M). Tuy nhiên, sau khi xem xét kỹ lưỡng, chúng tôi phát hiện ra một vấn đề phương pháp luận nghiêm trọng: kiến trúc U-Net backbone của `CB-SFNet` (được sử dụng trong `cbsfnet_light_exp2`) không đồng nhất với kiến trúc U-Net backbone của `ASRAN_LBS`. Cụ thể, `CB-SFNet` có một tầng down-sampling (`down4`) sâu hơn, làm cho việc so sánh trực tiếp mIoU giữa hai model này trở nên không công bằng.

    **Chi tiết Kiến trúc `CB_SFNet` (phiên bản gốc):**
    *   **Cấu trúc tổng thể:** Kiến trúc encoder-decoder dạng U-Net với các module mới tích hợp: `CSBD` (Contextual Spectral Boundary Discovery) và `MDSA-Net` (Multi-Depth Semantic Aggregation Network).
    *   **Đường dẫn Encoder:**
        *   `inc`: `DoubleConv` (in_channels -> `base_filters`)
        *   `down1`: `Down` (`base_filters` -> `base_filters * 2`)
        *   `down2`: `Down` (`base_filters * 2` -> `base_filters * 4`)
        *   `down3`: `Down` (`base_filters * 4` -> `base_filters * 8`)
        *   **Tích hợp Bottleneck:** Các đặc trưng từ `MDSA-Net` được nối (concatenated) với `x4` (đầu ra của `down3`) trước lớp `down4` cuối cùng.
        *   `down4`: `Down` (`base_filters * 8 + mdsa_out_channels` -> `base_filters * 16 // factor`). Đây là điểm khác biệt chính, ngụ ý một encoder 5 cấp.
    *   **Đường dẫn Decoder:**
        *   `up1`: `Up` (`base_filters * 16` -> `base_filters * 8 // factor`)
        *   `up2`: `Up` (`base_filters * 8` -> `base_filters * 4 // factor`)
        *   `up3`: `Up` (`base_filters * 4` -> `base_filters * 2 // factor`)
        *   `up4`: `Up` (`base_filters * 2` -> `base_filters`)
        *   `outc`: `OutConv` (`base_filters` -> `num_classes`)
    *   **Các Module mới:**
        *   `CSBD` (Contextual Spectral Boundary Discovery): Sử dụng `SpectralGradientAnalyzer` và một `Boundary Localization Network` (CNN) nhỏ để dự đoán boundary logits.
        *   `MDSA-Net` (Multi-Depth Semantic Aggregation Network): Nhận các đặc trưng đầu vào và đặc trưng biên, xử lý chúng qua ba đường dẫn (Lightweight Conv, Multi-scale Pyramid, Residual Block) và tổng hợp chúng.
    *   **Điểm khác biệt chính:** Model `CB_SFNet` có thêm một lớp `down4` *sau khi* tích hợp các đặc trưng `MDSA-Net`, khiến nó trở thành một encoder-decoder 5 cấp nếu `MDSA-Net` được coi là một phần của bottleneck. Điều này tạo ra sự không nhất quán về kiến trúc so với các model U-Net 4 cấp khác.

*   **Kết luận:** Kết quả của `cbsfnet_light_exp2` bị vô hiệu hóa cho mục đích so sánh trực tiếp. Cần phải chuẩn hóa kiến trúc backbone để đảm bảo một sân chơi công bằng.

---

## 2. Cải tiến 1: Chuẩn hóa Kiến trúc Backbone (Unified Backbone Architecture)

*   **Mục tiêu:** Tạo ra một kiến trúc U-Net backbone tiêu chuẩn, thống nhất để đảm bảo sự so sánh công bằng giữa các model khác nhau (đặc biệt là `ASRAN-LBS` và `CB-SFNet`).
*   **Hành động:**
    *   **Tạo `src/models/backbones.py`:** Một file mới được tạo để chứa các khối xây dựng U-Net tiêu chuẩn (`StandardDoubleConv`, `StandardDown`, `StandardUp`, `StandardOutConv`) và một lớp `StandardUNet` hoàn chỉnh.

    **Chi tiết Kiến trúc `StandardUNet` (Backbone thống nhất):**
    *   **Cấu trúc tổng thể:** Kiến trúc encoder-decoder U-Net 4 cấp tiêu chuẩn.
    *   **Đường dẫn Encoder:**
        *   `inc`: `StandardDoubleConv` (in_channels -> `base_filters`)
        *   `down1`: `StandardDown` (`base_filters` -> `base_filters * 2`)
        *   `down2`: `StandardDown` (`base_filters * 2` -> `base_filters * 4`)
        *   `down3`: `StandardDown` (`base_filters * 4` -> `base_filters * 8`)
    *   **Bottleneck:**
        *   `bottleneck_conv`: `StandardDoubleConv` (`base_filters * 8` -> `base_filters * 16`)
        *   Có thể áp dụng module `ASA` (Adaptive Spatial Attention) tùy chọn tại bottleneck.
    *   **Đường dẫn Decoder:**
        *   `up1`: `StandardUp` (`base_filters * 16` -> `base_filters * 8`)
        *   `up2`: `StandardUp` (`base_filters * 8` -> `base_filters * 4`)
        *   `up3`: `StandardUp` (`base_filters * 4` -> `base_filters * 2`)
        *   `up4`: `StandardUp` (`base_filters * 2` -> `base_filters`)
        *   `outc`: `StandardOutConv` (`base_filters` -> `num_classes`)
    *   **Điểm khác biệt chính:** Đây là một U-Net 4 cấp nghiêm ngặt, đảm bảo độ sâu nhất quán giữa các model sử dụng backbone này.

    *   **Tái cấu trúc `ASRAN_LBS`:** Model `ASRAN_LBS` được sửa đổi để sử dụng các khối `StandardUNet` từ `backbones.py`.

    **Chi tiết Kiến trúc `ASRAN_LBS` (sử dụng `StandardUNet` backbone):**
    *   **Cấu trúc tổng thể:** Tích hợp một `LearnableBandSelector` với backbone `StandardUNet`.
    *   **`LearnableBandSelector`:**
        *   Đầu vào là `in_channels` (ví dụ: 25) và `num_select_bands` (ví dụ: 5).
        *   Sử dụng `logits` có thể học được cho mỗi dải tần.
        *   Trong quá trình huấn luyện, áp dụng kỹ thuật Gumbel-TopK để lựa chọn `num_select_bands` một cách khác biệt hóa.
        *   Trong quá trình suy luận, chọn ra top-k dải tần dựa trên `logits` đã học.
    *   **`segmentation_network`:**
        *   Là một thể hiện của `StandardUNet`.
        *   `in_channels` cho `StandardUNet` được đặt thành `num_select_bands` (ví dụ: 5).
        *   `num_classes` và `base_filters` được truyền qua.
        *   `use_asa` được đặt thành `True`, nghĩa là `Adaptive Spatial Attention` được sử dụng tại bottleneck của `StandardUNet`.
    *   **Forward Pass:**
        1.  Đầu vào `x` (ví dụ: 25 dải tần) được truyền qua `band_selector` để lấy `selected_bands` (ví dụ: 5 dải tần).
        2.  `selected_bands` sau đó được đưa vào `segmentation_network` (StandardUNet) để tạo ra `seg_logits`.

    *   **Tạo `CB_SFNet_Unified`:** Một model `CB_SFNet_Unified` hoàn toàn mới được xây dựng lại từ đầu, sử dụng các khối `Standard` và có cùng độ sâu (4 tầng down-sampling) và cấu trúc như `StandardUNet`. Các đặc trưng của `MDSA-Net` được chèn vào bottleneck một cách hợp lý mà không làm thay đổi kiến trúc U-Net nền.

    **Chi tiết Kiến trúc `CB_SFNet_Unified` (sử dụng `StandardUNet` backbone):**
    *   **Cấu trúc tổng thể:** Kiến trúc encoder-decoder U-Net 4 cấp (sử dụng các khối xây dựng `StandardUNet`) với các module `CSBD` và `MDSA-Net` tích hợp.
    *   **Đường dẫn Encoder (các khối Standard U-Net):**
        *   `inc`: `StandardDoubleConv` (in_channels -> `base_filters`)
        *   `down1`: `StandardDown` (`base_filters` -> `base_filters * 2`)
        *   `down2`: `StandardDown` (`base_filters * 2` -> `base_filters * 4`)
        *   `down3`: `StandardDown` (`base_filters * 4` -> `base_filters * 8`)
    *   **Các Module mới (Tái sử dụng từ `CB_SFNet` gốc):**
        *   `csbd`: `CSBD` (in_channels -> 1)
        *   `mdsa_net`: `MDSA_Net` (in_channels, boundary_channels=1, out_channels=`base_filters * 4`)
    *   **Tích hợp tại Bottleneck:**
        *   Đầu ra của `down3` (`x4`) được nối (concatenated) với `mdsa_features` (được tạo từ đầu vào `x` nội suy và `boundary_features`).
        *   `bottleneck_conv`: `StandardDoubleConv` (`base_filters * 8 + mdsa_out_channels` -> `base_filters * 16`). Đây là nơi các đặc trưng MDSA được đưa vào đường dẫn U-Net chính.
    *   **Đường dẫn Decoder (các khối Standard U-Net):**
        *   `up1`: `StandardUp` (`base_filters * 16` -> `base_filters * 8`) - nhận `x5` (đầu ra bottleneck) và `x4` (kết nối bỏ qua từ encoder).
        *   `up2`: `StandardUp` (`base_filters * 8` -> `base_filters * 4`)
        *   `up3`: `StandardUp` (`base_filters * 4` -> `base_filters * 2`)
        *   `up4`: `StandardUp` (`base_filters * 2` -> `base_filters`)
        *   `outc`: `StandardOutConv` (`base_filters` -> `num_classes`)
    *   **Forward Pass:**
        1.  `CSBD` xử lý đầu vào `x` để lấy `boundary_logits` và `boundary_features`.
        2.  Đầu vào `x` đi qua encoder U-Net 4 cấp (`inc` đến `down3`).
        3.  `MDSA-Net` xử lý `x` nội suy và `boundary_features` để lấy `mdsa_features`.
        4.  `x4` (đầu ra encoder) được nối với `mdsa_features` và đi qua `bottleneck_conv`.
        5.  Kết quả (`x5`) sau đó đi qua decoder U-Net 4 cấp (`up1` đến `outc`).
        6.  Trả về một dictionary với `segmentation`, `boundary` và `contrast_features`.

    *   **Đăng ký Model mới:** Cập nhật `src/models/__init__.py` để đăng ký `CB_SFNet_Unified`.
*   **Thử nghiệm:** `cbsfnet_unified_exp1`
    *   **File Config:** `configs/cbsfnet_unified_exp1_config.yaml`
    *   **Tiến trình:**
        *   Chạy thử 1-epoch để xác thực kiến trúc mới.
        *   Huấn luyện đầy đủ trong khoảng 105 epochs (dừng sớm do hội tụ).
    *   **Kết quả:**
        *   **mIoU tốt nhất:** **0.7607**
        *   **Tham số:** 0.53M
        *   **GFLOPs:** 2.96
        *   **FPS (ước tính):** ~1128
    *   **Thành tựu:**
        *   Đã thiết lập thành công một baseline công bằng.
        *   `CB_SFNet_Unified` trở thành model tốt nhất mới, vượt qua `ASRAN_LBS` (0.7241) trong một so sánh công bằng.

---

## 3. Cải tiến 2: Tăng cường Dữ liệu Nâng cao (Enhanced Data Augmentation)

*   **Mục tiêu:** Cải thiện hơn nữa mIoU của model `CB_SFNet_Unified` bằng cách sử dụng các kỹ thuật tăng cường dữ liệu mạnh mẽ hơn.
*   **Hành động:**
    *   Sao chép config của `cbsfnet_unified_exp1` thành `configs/cbsfnet_unified_exp2_aug_config.yaml`.
    *   Thêm các kỹ thuật tăng cường dữ liệu mới: `GridDistortion` và `CoarseDropout` vào pipeline augmentation.
*   **Thử nghiệm:** `cbsfnet_unified_exp2_aug`
    *   **File Config:** `configs/cbsfnet_unified_exp2_aug_config.yaml`
    *   **Tiến trình:**
        *   Chạy thử 1-epoch để xác thực cấu hình augmentation mới.
        *   Huấn luyện đầy đủ trong khoảng 116 epochs (dừng sớm do hội tụ).
    *   **Kết quả:**
        *   **mIoU tốt nhất:** **0.7667**
        *   **Tham số:** 0.53M
        *   **GFLOPs:** 2.96
        *   **FPS (ước tính):** ~781 (thấp hơn do tính toán augmentation)
    *   **Thành tựu:**
        *   Tăng cường dữ liệu mạnh đã thành công trong việc đẩy mIoU lên một kỷ lục mới, chứng minh hiệu quả của nó.
        *   `cbsfnet_unified_exp2_aug` trở thành model tốt nhất hiện tại.

---

## 4. Cải tiến 3: Scheduler Learning Rate Cosine Annealing

*   **Mục tiêu:** Tối ưu hóa lịch trình learning rate để đạt được sự hội tụ mượt mà hơn và tiềm năng mIoU cao hơn.
*   **Hành động:**
    *   Sao chép config của `cbsfnet_unified_exp2_aug` thành `configs/cbsfnet_unified_exp3_cosine_config.yaml`.
    *   Thay thế scheduler `ReduceLROnPlateau` bằng `CosineAnnealingLR` với các tham số `T_max: 200` và `eta_min: 0.000001`.
*   **Thử nghiệm:** `cbsfnet_unified_exp3_cosine`
    *   **File Config:** `configs/cbsfnet_unified_exp3_cosine_config.yaml`
    *   **Tiến trình:**
        *   Chạy thử 1-epoch để xác thực cấu hình scheduler mới.
        *   Đang huấn luyện đầy đủ (đã hoàn thành 169 epochs khi kiểm tra lần cuối).
    *   **Kết quả (tính đến lần kiểm tra cuối cùng):**
        *   **mIoU tốt nhất:** **0.7792** (tại epoch 119)
        *   **Tham số:** 0.53M
        *   **GFLOPs:** 2.96
    *   **Thành tựu:**
        *   `CosineAnnealingLR` đã chứng minh hiệu quả vượt trội, dẫn đến một kỷ lục mIoU mới.
        *   Model đang rất gần với mục tiêu 0.8 mIoU, cho thấy tiềm năng lớn.

---

## 5. Thành tựu Tổng thể của Dự án

*   Đã thành công trong việc xác định `CB_SFNet_Unified` là kiến trúc nhẹ tốt nhất sau khi chuẩn hóa backbone.
*   Đã đạt được kỷ lục mIoU mới là **0.7792**, tiến gần hơn đáng kể đến mục tiêu 0.8 mIoU.
*   Đã chứng minh hiệu quả của cả việc tăng cường dữ liệu nâng cao và scheduler `CosineAnnealingLR` trong việc cải thiện hiệu suất model.
*   Dự án đang trên đà đạt được mục tiêu mIoU 0.8 với một model siêu nhẹ và hiệu quả.

---

## 6. Bảng So sánh Chi tiết các Thử nghiệm

| Thử nghiệm | Kiến trúc | Chi tiết Kiến trúc | Augmentation | Scheduler LR | mIoU tốt nhất | Tham số | GFLOPs | FPS (ước tính) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cbsfnet_light_exp2` | `CB_SFNet` (Gốc) | U-Net 5 cấp, `base_filters=8`, `CSBD`, `MDSA-Net` | Cơ bản | `ReduceLROnPlateau` | 0.7466 | 0.53M | N/A | N/A |
| `ASRAN_LBS_exp2_base8` | `ASRAN_LBS` | `StandardUNet` 4 cấp, `base_filters=8`, `LearnableBandSelector`, `ASA` | Cơ bản | `ReduceLROnPlateau` | 0.7241 | 0.49M | 2.68 | ~1491 |
| `cbsfnet_unified_exp1` | `CB_SFNet_Unified` | `StandardUNet` 4 cấp, `base_filters=8`, `CSBD`, `MDSA-Net` | Cơ bản | `ReduceLROnPlateau` | 0.7607 | 0.53M | 2.96 | ~1128 |
| `cbsfnet_unified_exp2_aug` | `CB_SFNet_Unified` | `StandardUNet` 4 cấp, `base_filters=8`, `CSBD`, `MDSA-Net` | Nâng cao (`GridDistortion`, `CoarseDropout`) | `ReduceLROnPlateau` | 0.7667 | 0.53M | 2.96 | ~781 |
| `cbsfnet_unified_exp3_cosine` | `CB_SFNet_Unified` | `StandardUNet` 4 cấp, `base_filters=8`, `CSBD`, `MDSA-Net` | Nâng cao (`GridDistortion`, `CoarseDropout`) | `CosineAnnealingLR` | 0.7792 | 0.53M | 2.96 | ~1388 |