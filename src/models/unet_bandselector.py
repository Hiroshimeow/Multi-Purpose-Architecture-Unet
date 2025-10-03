"""
Kiến trúc UnetBandS (UNet with BandSelector)

   * Parameters: in_channels=25, reduced_depth=3, initial_filters=32, num_classes=5,
     bilinear=False.

   * Architecture Flow:
       1. Input: (N, 25, H, W)
       2. Preprocessing: BandSelector(25, 3) processes the Input -> preprocessed_output (shape
          (N, 3, H, W))
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
from .attention_modules import BandSelector

class UnetBandS(nn.Module):
    def __init__(self, in_channels, num_classes, num_selected_bands=5, **kwargs):
        super().__init__()
        # Updated to use the new parameter names for LearnableBandSelector
        self.band_selector = BandSelector(in_channels=in_channels, num_bands_to_select=num_selected_bands)
        self.unet = UNetBase(
            in_channels=num_selected_bands, 
            num_classes=num_classes,
            attention_block=None,
            **kwargs
        )

    def forward(self, x):
        selected_bands = self.band_selector(x)
        # The trainer now expects a dictionary output
        segmentation_output = self.unet(selected_bands)
        return {'segmentation': segmentation_output}
