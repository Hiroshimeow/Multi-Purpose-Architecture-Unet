# src/models/gemini_unet_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Band_Selector import BandSelector # Import the new BandSelector

# --- WangTeacher's Code --- 
class SE_attention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_attention, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # This attention is designed for 3D data (N, C, D, H, W)
        # The forward pass from the teacher's code seems to assume (N, C, D) after pooling
        # Let's adapt it for the 5D tensor from SAC
        y = x.mean(dim=[3,4], keepdim=True) # Pool spatially, keep channel and depth
        y = y.squeeze(-1).squeeze(-1) # Shape: (N, C, D)
        
        # Conv1d expects (N, C, Length), so we permute
        y = y.permute(0, 2, 1) # Shape: (N, D, C)
        # This is not right. Conv1d is on the channel dimension.
        # Let's stick to the original implementation from the teacher, assuming the input is (N, C, D)
        # The input to forward is (N, C, D, H, W)
        y = x.mean(dim=[3, 4]) # -> (N, C, D)
        # The teacher's code has a view, which is not robust. Let's assume Conv1d works on the last dim.
        # It doesn't. It works on the last dim if kernel_size=1. It works on the channel dim.
        # The teacher's code is a bit ambiguous. Let's assume the intention is channel attention over depth.
        # y = y.view(x.size(0), x.size(1), x.size(2)) # This is (N, C, D)
        y = self.fc1(y) # Input (N, C, D), expects (N, C, L)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(-1).unsqueeze(-1) # -> (N, C, D, 1, 1)
        return x * y

class SAC(nn.Module):
    def __init__(self, input_depth=25, reduced_depth=3, channels=32, use_sea=True):
        super(SAC, self).__init__()
        self.use_sea = use_sea
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )
        if self.use_sea:
            self.attention = SE_attention(channels, reduction=4)
        
        kernel_depth = input_depth - reduced_depth + 1
        self.reduce_conv = nn.Conv3d(channels, 1, kernel_size=(kernel_depth,1,1))
        
        self.skip_conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(1),
            nn.ReLU()
        )
        self.down_conv = nn.Conv3d(1, 1, kernel_size=(kernel_depth,1,1))

    def forward(self, x):
        # Input x is (N, C, H, W), where C is spectral channels
        # We need to treat spectral as depth for 3D conv
        x_in = x.unsqueeze(1) # -> (N, 1, C, H, W)
        
        x_branch = self.conv1(x_in)
        x_branch = self.conv2(x_branch)
        if self.use_sea:
            x_branch = self.attention(x_branch)
        x_branch = self.reduce_conv(x_branch)
        
        x_skip = self.skip_conv(x_in)
        x_skip = self.down_conv(x_skip)
        
        out = F.relu(x_branch + x_skip)
        # Output is (N, 1, reduced_depth, H, W)
        return out.squeeze(1) # -> (N, reduced_depth, H, W)

# --- Components from GeminiUNet --- 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x)); return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__(); self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True); max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1); x = self.conv1(x); return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__(); self.ca = ChannelAttention(in_planes, ratio); self.sa = SpatialAttention(kernel_size)
    def forward(self, x): x = x * self.ca(x); x = x * self.sa(x); return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(); self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.conv(x)

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

class GeminiUNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, initial_filters=32, depth=4, dropout_rate=0.2, bilinear=True, sac_params=None, use_bcd=False, bcd_intermediate_channels=3, band_selector_config=None, **kwargs):
        super(GeminiUNetV2, self).__init__()
        self.depth = depth
        self.use_sac = sac_params is not None
        self.use_bcd = use_bcd
        self.use_band_selector = band_selector_config is not None

        # --- Band Selector --- 
        if self.use_band_selector:
            original_in_channels = in_channels # Store original in_channels for BandSelector
            self.band_selector = BandSelector(**band_selector_config)
            unet_input_channels_after_preprocessing = band_selector_config['num_selected_bands']
        else:
            self.band_selector = None
            unet_input_channels_after_preprocessing = in_channels

        if self.use_bcd:
            self.bcd = BandCompressionDecompression(unet_input_channels_after_preprocessing, bcd_intermediate_channels)
            # BCD outputs original channel count, which is now the output of band_selector or original in_channels
            unet_input_channels_after_preprocessing = unet_input_channels_after_preprocessing 
        else:
            unet_input_channels_after_preprocessing = unet_input_channels_after_preprocessing # If BCD not used, input channels remain original

        if self.use_sac:
            # FIX: Truyền đúng in_channels vào cho SAC
            sac_params['input_depth'] = unet_input_channels_after_preprocessing
            self.sac = SAC(**sac_params)
            unet_in_channels = sac_params.get('reduced_depth', 3)
        else:
            unet_in_channels = unet_input_channels_after_preprocessing # If SAC not used, channels remain as they are after BCD or original

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        in_ch = unet_in_channels
        for i in range(depth):
            out_ch = initial_filters * (2**i)
            self.encoders.append(nn.Sequential(DoubleConv(in_ch, out_ch), CBAM(out_ch), nn.MaxPool2d(2), nn.Dropout(dropout_rate)))
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(DoubleConv(in_ch, in_ch * 2), CBAM(in_ch * 2))
        in_ch *= 2

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_ch = initial_filters * (2**i)
            if bilinear:
                up_module = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, up_ch, kernel_size=1))
            else:
                up_module = nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2)
            
            self.decoders.append(nn.ModuleDict({'up': up_module, 'conv': nn.Sequential(DoubleConv(up_ch * 2, up_ch), CBAM(up_ch))}))
            in_ch = up_ch

        self.outc = OutConv(initial_filters, num_classes)

    def forward(self, x):
        if self.use_band_selector:
            x = self.band_selector(x)
        if self.use_bcd:
            x = self.bcd(x)
        if self.use_sac: # Apply SAC after BCD if both are enabled
            x = self.sac(x)
        
        skip_connections = []

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
            x = self.decoders[i]['up'](x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[i]['conv'](x)

        return self.outc(x)
