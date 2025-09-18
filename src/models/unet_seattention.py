"""
Kiến trúc Unet-SEAttention (SAC)

   * Parameters: in_channels=25, reduced_depth=3, initial_filters=32, num_classes=5,
     bilinear=False.

   * Architecture Flow:
       1. Input: (N, 25, H, W)
       2. Preprocessing: SAC(in_depth=25, reduced_depth=3) processes the Input ->
          preprocessed_output (shape (N, 3, H, W))
       3. Inc: DoubleConv(3, 32) processes preprocessed_output -> inc_output
       4. Encoder 1: Down(32, 64) processes inc_output -> encoder1_output
       5. Encoder 2: Down(64, 128) processes encoder1_output -> encoder2_output
       6. Encoder 3: Down(128, 256) processes encoder2_output -> encoder3_output
       7. Bottleneck: Down(256, 512) processes encoder3_output -> bottleneck_output
       8. Decoder 1: Up(512, 256, 256) receives bottleneck_output and encoder3_output ->
          decoder1_output
       9. Decoder 2: Up(256, 128, 128) receives decoder1_output and encoder2_output ->
          decoder2_output
       10. Decoder 3: Up(128, 64, 64) receives decoder2_output and encoder1_output ->
           decoder3_output
       11. Decoder 4: Up(64, 32, 32) receives decoder3_output and inc_output -> decoder4_output
       12. Output: OutConv(32, 5) receives decoder4_output -> final_output (shape (N, 5, H, W))
"""
import torch.nn as nn
from .unet_base import UNetBase
from .attention_modules import SAC

class UnetSEAttention(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        # SAC giảm số kênh từ 25 xuống 5
        self.sac = SAC(input_depth=in_channels, reduced_depth=5, use_sea=True)
        self.unet = UNetBase(
            in_channels=5, # UNet giờ nhận 5 kênh đầu vào
            num_classes=num_classes,
            attention_block=None, # Không có attention block bên trong UNet
            **kwargs
        )

    def forward(self, x):
        x = self.sac(x)
        return self.unet(x)
