# src/models/gemini_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Components from geminii_learnable_unet.py ---

class BandSelector(nn.Module):
    def __init__(self, in_channels):
        super(BandSelector, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.attention(x)
        return x * weights, weights.squeeze()

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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        return self.conv(x)

class GeminiUNet(nn.Module):
    def __init__(self, in_channels, num_classes, initial_filters=64, depth=5, dropout_rate=0.3, **kwargs):
        super(GeminiUNet, self).__init__()
        self.depth = depth
        self.band_selector = BandSelector(in_channels)

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = initial_filters * (2**i)
            self.encoders.append(
                nn.Sequential(
                    DoubleConv(in_ch, out_ch),
                    CBAM(out_ch),
                    nn.MaxPool2d(2),
                    nn.Dropout(dropout_rate)
                )
            )
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            DoubleConv(in_ch, in_ch * 2),
            CBAM(in_ch * 2)
        )
        in_ch *= 2

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        self.deep_supervision_outputs = nn.ModuleList()
        # Add DS for the last 2 decoder stages by default
        self.ds_stages = 2 

        for i in range(depth - 1, -1, -1):
            up_ch = initial_filters * (2**i)
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2),
                    DoubleConv(up_ch * 2, up_ch), # Skip connection doubles channels
                    CBAM(up_ch)
                )
            )
            in_ch = up_ch
            if i >= depth - self.ds_stages:
                 self.deep_supervision_outputs.append(OutConv(up_ch, num_classes))

        self.outc = OutConv(initial_filters, num_classes)

    def forward(self, x):
        x, band_weights = self.band_selector(x)
        skip_connections = []
        ds_outputs = []

        # Encoder path
        for i in range(self.depth):
            x = self.encoders[i][0](x) # DoubleConv
            x = self.encoders[i][1](x) # CBAM
            skip_connections.append(x)
            x = self.encoders[i][2](x) # MaxPool
            x = self.encoders[i][3](x) # Dropout

        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for i in range(self.depth):
            x = self.decoders[i][0](x) # ConvTranspose
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[i][1](x) # DoubleConv
            x = self.decoders[i][2](x) # CBAM

            if i < self.ds_stages:
                ds_out = self.deep_supervision_outputs[i](x)
                ds_outputs.append(ds_out)

        main_output = self.outc(x)
        
        # The outputs are naturally from highest res to lowest, reverse them
        # to match the logic in the loss function (main output + high-res DS first)
        ds_outputs.reverse()

        return [main_output] + ds_outputs, band_weights