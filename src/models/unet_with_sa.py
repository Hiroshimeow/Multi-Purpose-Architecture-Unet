# src/models/unet_with_sa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_blocks import DoubleConv, Down, Up, OutConv

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

class UNetWithSA(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True, base_filters=64):
        super(UNetWithSA, self).__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, (base_filters * 16) // factor)
        
        self.sa = SpatialAttention()
        
        self.up1 = Up((base_filters * 16) // factor, base_filters * 8, (base_filters * 8) // factor, bilinear)
        self.up2 = Up((base_filters * 8) // factor, base_filters * 4, (base_filters * 4) // factor, bilinear)
        self.up3 = Up((base_filters * 4) // factor, base_filters * 2, (base_filters * 2) // factor, bilinear)
        self.up4 = Up((base_filters * 2) // factor, base_filters, base_filters, bilinear)
        self.outc = OutConv(base_filters, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.sa(x5) * x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
