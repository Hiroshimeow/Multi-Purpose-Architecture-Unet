"""
Kiến trúc UnetDeepSupervision

Đây là một phiên bản UNet tích hợp kỹ thuật "Deep Supervision". Thay vì chỉ có một
output ở cuối cùng, model này có nhiều output ở các tầng decoder khác nhau.

Luồng dữ liệu chi tiết:
- **Input**: `(N, C, H, W)`
- **Kiến trúc**: Tương tự UNet tiêu chuẩn, nhưng có sự thay đổi ở đầu ra.
- **Decoder Path & Outputs**:
    - Sau mỗi khối `DoubleConv` ở các tầng decoder cuối cùng (ví dụ 2 tầng cuối),
      một lớp `OutConv` được thêm vào để tạo ra một ảnh segmentation phụ.
    - Model sẽ trả về một **list** các tensor:
      `[main_output, ds_output_1, ds_output_2, ...]`
    - `main_output` là output chính có độ phân giải đầy đủ.
    - `ds_output_x` là các output phụ có độ phân giải thấp hơn.
- **Mục đích**: Cung cấp tín hiệu loss trực tiếp cho các lớp nông hơn, giúp quá trình
  huấn luyện ổn định và hiệu quả hơn.

*Lưu ý: Model này không sử dụng `UNetBase` vì cấu trúc `forward` và dạng output khác biệt.*
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_blocks import DoubleConv, OutConv

class UnetDeepSupervision(nn.Module):
    def __init__(self, in_channels, num_classes, initial_filters=64, depth=5, **kwargs):
        super().__init__()
        self.depth = depth
        self.ds_stages = 2 # Số tầng decoder có deep supervision output

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = initial_filters * (2**i)
            self.encoders.append(
                nn.Sequential(DoubleConv(in_ch, out_ch), nn.MaxPool2d(2))
            )
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(in_ch, in_ch * 2)
        in_ch *= 2

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        self.deep_supervision_outputs = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            up_ch = initial_filters * (2**i)
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2),
                    DoubleConv(up_ch * 2, up_ch) # Skip connection doubles channels
                )
            )
            in_ch = up_ch
            # Thêm DS output cho các tầng cuối
            if i >= depth - self.ds_stages:
                 self.deep_supervision_outputs.append(OutConv(up_ch, num_classes))

        self.outc = OutConv(initial_filters, num_classes)

    def forward(self, x):
        skip_connections = []
        ds_outputs = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder[0](x) # DoubleConv
            skip_connections.append(x)
            x = encoder[1](x) # MaxPool

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

            # Lấy output từ các tầng DS
            if i < self.ds_stages:
                ds_out = self.deep_supervision_outputs[i](x)
                ds_outputs.append(ds_out)

        main_output = self.outc(x)
        
        # Đảo ngược list để output có độ phân giải cao nhất ở đầu
        ds_outputs.reverse()

        return [main_output] + ds_outputs