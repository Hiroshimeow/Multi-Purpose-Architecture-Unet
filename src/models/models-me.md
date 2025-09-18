# Giải thích Kiến trúc Model

Đây là tài liệu giải thích cấu trúc của các model UNet đã được module hóa trong thư mục `src/models`.

## 1. Các Khối Xây dựng Cơ bản (`base_blocks.py`)

- **`DoubleConv`**: Khối convolution kép, là xương sống của UNet.
    ```
    Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d -> ReLU
    ```
- **`Down`**: Khối giảm kích thước trong Encoder.
    ```
    MaxPool2d(2x2) -> DoubleConv
    ```
- **`Up`**: Khối tăng kích thước trong Decoder.
    ```
    Upsample (Bilinear) hoặc ConvTranspose2d -> Concat với Skip Connection -> DoubleConv
    ```

## 2. Các Module Kỹ thuật (`attention_modules.py`)

Đây là các "kỹ thuật" đặc biệt có thể được gắn vào kiến trúc UNet.

- **`CBAM`**: Giúp model tập trung vào cả những vùng (spatial) và những kênh (channel) quan trọng.
- **`SAC`**: Giảm số lượng kênh quang phổ (ví dụ từ 25 xuống 5) một cách "thông minh" bằng 3D convolution.
- **`BandSelector`**: Học cách kết hợp các kênh đầu vào để tạo ra một số lượng kênh mới ít hơn.
- **`BCD`**: Tạo một "nút cổ chai thông tin" (information bottleneck) cho các kênh bằng 2 lớp Conv 1x1.

---

## 3. Phân tích Kiến trúc Chi tiết

### 3.1. `UNetBase`

- **Mô tả**: Kiến trúc UNet xương sống, không chứa module đặc biệt nào.
- **Luồng dữ liệu**:
    1.  **Input**: `(N, C, H, W)`
    2.  **Inc**: `DoubleConv(C, 64)` -> `skip1`
    3.  **Encoder 1**: `Down(64, 128)` -> `skip2`
    4.  **Encoder 2**: `Down(128, 256)` -> `skip3`
    5.  **Encoder 3**: `Down(256, 512)` -> `skip4`
    6.  **Bottleneck**: `Down(512, 1024)` -> `x`
    7.  **Decoder 1**: `Up` nhận `x` và `skip4` -> `x`
    8.  **Decoder 2**: `Up` nhận `x` và `skip3` -> `x`
    9.  **Decoder 3**: `Up` nhận `x` và `skip2` -> `x`
    10. **Decoder 4**: `Up` nhận `x` và `skip1` -> `x`
    11. **Output**: `OutConv` nhận `x` -> `(N, num_classes, H, W)`

### 3.2. `UnetCbam`

- **Mô tả**: `UNetBase` được tích hợp `CBAM`.
- **Luồng dữ liệu**:
    - Tương tự `UNetBase`, nhưng sau mỗi khối `DoubleConv` trong toàn bộ Encoder và Bottleneck, một khối `CBAM` được thêm vào.
    - **Ví dụ Encoder 1**:
        ```
        MaxPool2d -> DoubleConv(64, 128) -> CBAM(128) -> skip2
        ```

### 3.3. `UnetCbam + BandSelector` (Kiến trúc kết hợp)

- **Mô tả**: Sử dụng `BandSelector` để chọn lọc kênh đầu vào, sau đó đưa kết quả vào `UnetCbam`.
- **Luồng dữ liệu**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Preprocessing**: `BandSelector(25, 5)` -> `(N, 5, H, W)`
    3.  **UNet Core**: Tensor 5 kênh được đưa vào `UnetCbam` (với `in_channels=5`).
    4.  **Output**: `(N, num_classes, H, W)`

### 3.4. `UnetCbam + SEAttention` (Kiến trúc `UnetCbamSeattention`)

- **Mô tả**: Sử dụng `SAC` (chứa `SE_attention`) để giảm chiều quang phổ, sau đó đưa vào `UnetCbam`.
- **Luồng dữ liệu**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Preprocessing**: `SAC(input_depth=25, reduced_depth=5)` -> `(N, 5, H, W)`
    3.  **UNet Core**: Tensor 5 kênh được đưa vào `UnetCbam` (với `in_channels=5`). Kiến trúc này vừa có `SE_attention` trong `SAC` ở đầu vào, vừa có `CBAM` trong thân UNet.
    4.  **Output**: `(N, num_classes, H, W)`
