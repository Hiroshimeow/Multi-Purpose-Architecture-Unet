"""
Kiến trúc UnetDepthwiseSeparable

Đây là phiên bản UNet hiệu suất cao, thay thế toàn bộ các khối `DoubleConv`
truyền thống bằng `DoubleDSConv` (Depthwise Separable Convolution).

Luồng dữ liệu chi tiết:
- **Input**: `(N, C, H, W)`
- **Kiến trúc**: Vẫn là kiến trúc UNet hình chữ U, nhưng mọi nơi sử dụng
  `DoubleConv` (trong `inc`, `Down`, `Up`, `bottleneck`) đều được thay bằng
  `DoubleDSConv`. Điều này làm giảm đáng kể số lượng tham số và phép tính.
- **Output**: `(N, num_classes, H, W)`

*Lưu ý: Model này không sử dụng `UNetBase` vì nó thay đổi khối building block cốt lõi.*
"""
import torch
import torch.nn as nn
from .base_blocks import DoubleDSConv, OutConv, Up

# Model này không dùng UNetBase vì nó thay thế các khối DoubleConv cốt lõi
class UnetDepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True, initial_filters=64, depth=4, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.depth = depth

        # --- Encoder ---
        self.inc = DoubleDSConv(in_channels, initial_filters)
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            in_filters = initial_filters * (2**i)
            out_filters = in_filters * 2
            self.encoders.append(nn.Sequential(nn.MaxPool2d(2), DoubleDSConv(in_filters, out_filters)))

        # --- Bottleneck ---
        factor = 2 if bilinear else 1
        bottleneck_in = initial_filters * (2**(depth - 1))
        bottleneck_out = bottleneck_in * 2 // factor
        self.encoders.append(nn.Sequential(nn.MaxPool2d(2), DoubleDSConv(bottleneck_in, bottleneck_out)))

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_filters = initial_filters * (2**(i+1)) // factor
            in_skip = initial_filters * (2**i)
            out_filters = initial_filters * (2**i)
            
            self.decoders.append(nn.ModuleDict({
                'up': nn.ConvTranspose2d(in_filters, out_filters, kernel_size=2, stride=2) if not bilinear else 
                      nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_filters, out_filters, kernel_size=1)),
                'conv': DoubleDSConv(out_filters + in_skip, out_filters)
            }))

        self.outc = OutConv(initial_filters, num_classes)

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        # Pop bottleneck output, no skip
        x = skip_connections.pop()
        skip_connections = skip_connections[::-1]

        for i, decoder in enumerate(self.decoders):
            x = decoder['up'](x)
            skip = skip_connections[i]
            # Pad và concat
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            x = decoder['conv'](x)

        return self.outc(x)