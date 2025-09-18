# src/models/attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BandSelector ---
class BandSelector(nn.Module):
    def __init__(self, original_in_channels, num_selected_bands):
        super().__init__()
        self.selection_weights = nn.Parameter(torch.randn(num_selected_bands, original_in_channels) * 0.01)

    def forward(self, x):
        softmax_weights = torch.softmax(self.selection_weights, dim=1)
        selected_bands = torch.einsum('sc,bchw->bshw', softmax_weights, x)
        return selected_bands

# --- SAC (Spectral Attention Conv) with internal SE_attention ---
class SE_attention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_attention, self).__init__()
        # Sửa lại cho đúng với 3D Conv (N, C, D, H, W)
        # Conv1d hoạt động trên chiều C, nên ta cần (N, C, D)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x: (N, C, D, H, W)
        # Pool H, W -> (N, C, D)
        y = x.mean(dim=[3, 4])
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Reshape để nhân -> (N, C, D, 1, 1)
        y = y.unsqueeze(-1).unsqueeze(-1)
        return x * y

class SAC(nn.Module):
    def __init__(self, input_depth=25, reduced_depth=3, channels=32, use_sea=True):
        super(SAC, self).__init__()
        self.use_sea = use_sea
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        if self.use_sea:
            self.attention = SE_attention(channels, reduction=4)
        
        kernel_depth = input_depth - reduced_depth + 1
        self.reduce_conv = nn.Conv3d(channels, 1, kernel_size=(kernel_depth,1,1))
        
        self.skip_conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        self.down_conv = nn.Conv3d(1, 1, kernel_size=(kernel_depth,1,1))

    def forward(self, x):
        # Input x: (N, C, H, W), C là số kênh quang phổ
        x_in = x.unsqueeze(1) # -> (N, 1, C, H, W)
        
        x_branch = self.conv1(x_in)
        x_branch = self.conv2(x_branch)
        if self.use_sea:
            x_branch = self.attention(x_branch)
        x_branch = self.reduce_conv(x_branch)
        
        x_skip = self.skip_conv(x_in)
        x_skip = self.down_conv(x_skip)
        
        out = F.relu(x_branch + x_skip)
        return out.squeeze(1) # -> (N, reduced_depth, H, W)

# --- CBAM (Convolutional Block Attention Module) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- BCD (Band Compression Decompression) ---
class BandCompressionDecompression(nn.Module):
    def __init__(self, in_channels, intermediate_channels):
        super(BandCompressionDecompression, self).__init__()
        self.compress = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1)
        self.decompress = nn.Conv2d(intermediate_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.compress(x))
        x = self.decompress(x)
        return x
