# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Khối Attention ---
class SE_attention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_attention, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Input x shape: (N, C, H, W)
        # Squeeze: (N, C)
        y = x.mean(dim=[2, 3])
        # Unsqueeze for Conv1d: (N, C, 1)
        y = y.unsqueeze(-1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Reshape for multiplication: (N, C, 1, 1)
        y = y.unsqueeze(-1)
        return x * y

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

# --- U-Net Building Blocks (Giữ nguyên) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

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
        self.attention = SE_attention((base_filters * 16) // factor)
        
        self.up1 = Up((base_filters * 16), (base_filters * 8) // factor, bilinear)
        self.up2 = Up((base_filters * 8), (base_filters * 4) // factor, bilinear)
        self.up3 = Up((base_filters * 4), (base_filters * 2) // factor, bilinear)
        self.up4 = Up((base_filters * 2), base_filters, bilinear)
        self.outc = OutConv(base_filters, num_classes)

    def forward(self, x):
        x = self.band_selector(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention(x5) # Áp dụng attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True, base_filters=64): # Thêm base_filters
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, (base_filters * 16) // factor)
        self.up1 = Up((base_filters * 16), (base_filters * 8) // factor, bilinear)
        self.up2 = Up((base_filters * 8), (base_filters * 4) // factor, bilinear)
        self.up3 = Up((base_filters * 4), (base_filters * 2) // factor, bilinear)
        self.up4 = Up((base_filters * 2), base_filters, bilinear)
        self.outc = OutConv(base_filters, num_classes)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2)
        x = self.up4(x, x1); return self.outc(x)
# --- Model Factory (Cập nhật) ---
def get_model(name: str, params: dict):
    print(f"   Initializing model: {name}") # Thêm log để dễ debug
    if name == "HybridUNet":
        return HybridUNet(**params)
    elif name == "UNet": # << THÊM DÒNG NÀY
        return UNet(**params) # << VÀ DÒNG NÀY
    else:
        raise ValueError(f"Model '{name}' not recognized.")