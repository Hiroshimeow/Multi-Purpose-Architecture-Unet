# src/models/flash_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Components from GeminiUNetV2 ---
# These are kept the same as they are proven to be effective.
class SE_attention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_attention, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = x.mean(dim=[3, 4])
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
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
        x_in = x.unsqueeze(1)
        
        x_branch = self.conv1(x_in)
        x_branch = self.conv2(x_branch)
        if self.use_sea:
            x_branch = self.attention(x_branch)
        x_branch = self.reduce_conv(x_branch)
        
        x_skip = self.skip_conv(x_in)
        x_skip = self.down_conv(x_skip)
        
        out = F.relu(x_branch + x_skip)
        return out.squeeze(1)

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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(); self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.conv(x)

# --- New Flash Components ---

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Block.
    A more efficient alternative to standard convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleDSConv(nn.Module):
    """(Depthwise Separable Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_ds_conv(x)

# --- Main FlashUNet Model ---

class FlashUNet(nn.Module):
    def __init__(self, in_channels, num_classes, initial_filters=32, depth=4, dropout_rate=0.2, bilinear=True, sac_params=None, **kwargs):
        super(FlashUNet, self).__init__()
        self.depth = depth
        self.use_sac = sac_params is not None
        if self.use_sac:
            self.sac = SAC(input_depth=in_channels, **sac_params)
            unet_in_channels = sac_params.get('reduced_depth', 3)
        else:
            unet_in_channels = in_channels

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        in_ch = unet_in_channels
        for i in range(depth):
            out_ch = initial_filters * (2**i)
            # Using the new DoubleDSConv block
            self.encoders.append(nn.Sequential(DoubleDSConv(in_ch, out_ch), CBAM(out_ch), nn.MaxPool2d(2), nn.Dropout(dropout_rate)))
            in_ch = out_ch

        # --- Bottleneck ---
        # Using the new DoubleDSConv block
        self.bottleneck = nn.Sequential(DoubleDSConv(in_ch, in_ch * 2), CBAM(in_ch * 2))
        in_ch *= 2

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_ch = initial_filters * (2**i)
            if bilinear:
                up_module = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, up_ch, kernel_size=1))
            else:
                up_module = nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2)
            
            # Using the new DoubleDSConv block
            self.decoders.append(nn.ModuleDict({'up': up_module, 'conv': nn.Sequential(DoubleDSConv(up_ch * 2, up_ch), CBAM(up_ch))}))
            in_ch = up_ch

        self.outc = OutConv(initial_filters, num_classes)

    def forward(self, x):
        if self.use_sac:
            x = self.sac(x)
        
        skip_connections = []

        # Encoder path
        for i in range(self.depth):
            x = self.encoders[i][0](x) # DoubleDSConv
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
