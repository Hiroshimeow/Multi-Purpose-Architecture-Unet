"""
Kiến trúc UnetCbam

   * Parameters: in_channels=25, initial_filters=32, num_classes=5, bilinear=False.

   * Architecture Flow:
       1. Input: (N, 25, H, W)
       2. Inc: DoubleConv(25, 32) processes the Input, then through CBAM(32) ->
          inc_cbam_output
       3. Encoder 1: Down(32, 64) processes inc_cbam_output, then through CBAM(64) ->
          encoder1_cbam_output
       4. Encoder 2: Down(64, 128) processes encoder1_cbam_output, then through CBAM(128) ->
          encoder2_cbam_output
       5. Encoder 3: Down(128, 256) processes encoder2_cbam_output, then through CBAM(256) ->
          encoder3_cbam_output
       6. Bottleneck: Down(256, 512) processes encoder3_cbam_output, then through CBAM(512) ->
          bottleneck_cbam_output
       7. Decoder 1: Up(512, 256, 256) receives bottleneck_cbam_output and
          encoder3_cbam_output -> decoder1_output
       8. Decoder 2: Up(256, 128, 128) receives decoder1_output and encoder2_cbam_output ->
          decoder2_output
       9. Decoder 3: Up(128, 64, 64) receives decoder2_output and encoder1_cbam_output ->
          decoder3_output
       10. Decoder 4: Up(64, 32, 32) receives decoder3_output and inc_cbam_output ->
           decoder4_output
       11. Output: OutConv(32, 5) receives decoder4_output -> final_output (shape (N, 5, H, W))
"""
import torch.nn as nn
from .unet_base import UNetBase
from .attention_modules import CBAM

class UnetCbam(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        self.unet = UNetBase(
            in_channels=in_channels,
            num_classes=num_classes,
            attention_block=CBAM, # Tiêm CBAM vào UNetBase
            **kwargs
        )

    def forward(self, x):
        return self.unet(x)
