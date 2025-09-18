"""
Kiến trúc UnetCbamSeattention

Đây là model kết hợp sức mạnh của cả `SAC` và `CBAM`.

Luồng dữ liệu chi tiết:
1.  **Input**: `(N, 25, H, W)` - Tensor ảnh đầu vào với 25 kênh quang phổ.

2.  **Preprocessing**: `SAC(input_depth=25, reduced_depth=5)`
    - Input được đưa qua khối `SAC` để giảm số kênh xuống còn 5.
    - Output từ SAC là `(N, 5, H, W)`.

3.  **UNet Core**: `UNetBase(in_channels=5, attention_block=CBAM, ...)`
    - Tensor 5 kênh này sau đó được xử lý bởi một kiến trúc `UNetBase`.
    - **Quan trọng**: `UNetBase` này được tiêm `CBAM` vào mỗi khối `DoubleConv`
      của nó, kết hợp cả hai loại attention.

4.  **Output**: `(N, num_classes, H, W)`
"""
import torch.nn as nn
from .unet_base import UNetBase
from .attention_modules import SAC, CBAM

class UnetCbamSeattention(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        # SAC giảm số kênh từ 25 xuống 5
        self.sac = SAC(input_depth=in_channels, reduced_depth=5, use_sea=True)
        # UNet nhận 5 kênh và sử dụng CBAM bên trong
        self.unet = UNetBase(
            in_channels=5, 
            num_classes=num_classes,
            attention_block=CBAM, # Áp dụng CBAM
            **kwargs
        )

    def forward(self, x):
        x = self.sac(x)
        return self.unet(x)