"""
Kiến trúc UnetBcd (UNet with Band Compression Decompression)

Sử dụng khối `BCD` như một bộ lọc thông tin ở đầu vào trước khi đưa vào `UNetBase`.

Luồng dữ liệu chi tiết:
1.  **Input**: `(N, C, H, W)` - Tensor ảnh đầu vào.

2.  **Preprocessing**: `BandCompressionDecompression(in_channels=C, intermediate_channels=3)`
    - Input được nén xuống 3 kênh rồi giải nén ngược lại C kênh.
    - `(N, C, H, W) -> Conv(1x1) -> (N, 3, H, W) -> ReLU -> Conv(1x1) -> (N, C, H, W)`

3.  **UNet Core**: `UNetBase(in_channels=C, ...)`
    - Tensor C kênh đã qua "nút cổ chai" được xử lý bởi `UNetBase` tiêu chuẩn.

4.  **Output**: `(N, num_classes, H, W)`
"""
import torch.nn as nn
from .unet_base import UNetBase
from .attention_modules import BandCompressionDecompression

class UnetBcd(nn.Module):
    def __init__(self, in_channels, num_classes, intermediate_channels=3, **kwargs):
        super().__init__()
        self.bcd = BandCompressionDecompression(in_channels=in_channels, intermediate_channels=intermediate_channels)
        # UNet vẫn nhận in_channels vì BCD trả về số kênh ban đầu
        self.unet = UNetBase(
            in_channels=in_channels, 
            num_classes=num_classes,
            **kwargs
        )

    def forward(self, x):
        x = self.bcd(x)
        return self.unet(x)