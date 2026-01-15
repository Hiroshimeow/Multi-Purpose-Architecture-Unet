"""
This file defines standardized U-Net backbone architectures to be used across different
experimental models to ensure fair and consistent comparisons.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_modules import SpatialAttention

# ============================================================
# Spectral Reconstruction Head
# ============================================================

class SpectralReconstructionHead(nn.Module):
    """
    Predicts the original 25 spectral bands from the bottleneck features.
    Acts as a regularizer to ensure the bottleneck retains rich spectral info.
    """
    def __init__(self, in_channels: int, out_channels: int = 25):
        super().__init__()
        # As per the paper: 2x Conv3x3(BN+ReLU) -> Conv1x1
        self.recon_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1) # Final layer to get 25 channels
        )

    def forward(self, x, original_size):
        x = self.recon_conv(x)
        # Upsample to the original input size to calculate loss
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x

# ============================================================
# Standard U-Net Building Blocks
# ============================================================

class StandardDoubleConv(nn.Module):
    """Standard Double Convolutional Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class StandardDown(nn.Module):
    """Standard Down-sampling Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class StandardUp(nn.Module):
    """Standard Up-sampling Block using ConvTranspose2d"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = StandardDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class StandardOutConv(nn.Module):
    """Standard Output Convolution Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ============================================================
# Standard U-Net Backbone with Optional ASA and SR Head
# ============================================================

class StandardUNet(nn.Module):
    """
    A standard U-Net architecture that can be used as a common backbone.
    Includes options to enable:
    - Adaptive Spatial Attention (ASA) at the bottleneck.
    - Spectral Reconstruction (SR) head as an auxiliary task.
    """
    def __init__(self, in_channels, num_classes, base_filters=16, use_asa=True, use_sr_head=False, sr_out_channels=25):
        super().__init__()
        self.use_asa = use_asa
        self.use_sr_head = use_sr_head
        
        self.inc = StandardDoubleConv(in_channels, base_filters)
        self.down1 = StandardDown(base_filters, base_filters * 2)
        self.down2 = StandardDown(base_filters * 2, base_filters * 4)
        self.down3 = StandardDown(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        bottleneck_channels = base_filters * 16
        self.bottleneck_conv = StandardDoubleConv(base_filters * 8, bottleneck_channels)
        
        if self.use_asa:
            self.asa = SpatialAttention()
        
        if self.use_sr_head:
            self.sr_head = SpectralReconstructionHead(bottleneck_channels, sr_out_channels)
        
        self.up1 = StandardUp(bottleneck_channels, base_filters * 8)
        self.up2 = StandardUp(base_filters * 8, base_filters * 4)
        self.up3 = StandardUp(base_filters * 4, base_filters * 2)
        self.up4 = StandardUp(base_filters * 2, base_filters)
        self.outc = StandardOutConv(base_filters, num_classes)

    def forward(self, x):
        original_size = x.shape[2:]
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        bottleneck_features = self.bottleneck_conv(x4)
        if self.use_asa:
            attention_map = self.asa(bottleneck_features)
            x5 = bottleneck_features * attention_map
        else:
            x5 = bottleneck_features
        
        # Main segmentation path
        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        logits = self.outc(x_seg)
        
        # If SR head is enabled, compute reconstruction and return a dict
        if self.use_sr_head:
            recon_output = self.sr_head(x5, original_size)
            return {
                "segmentation": logits,
                "reconstruction": recon_output
            }
        
        # Otherwise, return only segmentation logits for backward compatibility
        return logits