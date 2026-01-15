# src/models/hybrid_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_blocks import DoubleConv, Down, Up, OutConv
from .unet_with_sa import SpatialAttention # Re-using SpatialAttention

# --- Khối chọn kênh từ Code của Thầy ---
class BandSelector(nn.Module):
    def __init__(self, in_channels, selected_channels):
        super().__init__()
        self.in_channels = in_channels
        self.selected_channels = selected_channels
        # Tạo một tham số có thể học được
        self.mask = nn.Parameter(torch.randn(in_channels))

    def forward(self, x):
        # Áp dụng softmax để có được trọng số cho mỗi kênh
        weights = torch.softmax(self.mask, dim=0)
        # Lấy top k kênh có trọng số cao nhất
        topk_indices = torch.topk(weights, self.selected_channels, sorted=False)[1]
        # Sắp xếp lại các chỉ số để đảm bảo thứ tự nhất quán
        topk_indices_sorted, _ = torch.sort(topk_indices)
        return x[:, topk_indices_sorted, :, :]

# --- Kiến trúc HybridUNet Mới ---
class HybridUNet(nn.Module):
    def __init__(self, in_channels, selected_channels, num_classes, bilinear=True, base_filters=64):
        super(HybridUNet, self).__init__()
        
        self.band_selector = BandSelector(in_channels, selected_channels)
        
        # U-Net Core
        self.inc = DoubleConv(selected_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, (base_filters * 16) // factor)
        
        # Thêm Attention vào lớp sâu nhất
        self.attention = SpatialAttention() # Changed to SpatialAttention
        
        self.up1 = Up((base_filters * 16) // factor, base_filters * 8, (base_filters * 8) // factor, bilinear)
        self.up2 = Up((base_filters * 8) // factor, base_filters * 4, (base_filters * 4) // factor, bilinear)
        self.up3 = Up((base_filters * 4) // factor, base_filters * 2, (base_filters * 2) // factor, bilinear)
        self.up4 = Up((base_filters * 2) // factor, base_filters, base_filters, bilinear)
        self.outc = OutConv(base_filters, num_classes)

    def forward(self, x):
        x = self.band_selector(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention(x5) * x5 # Áp dụng attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
