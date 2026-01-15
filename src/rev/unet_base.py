"""
Kiến trúc UNetBase (Xương sống)

   * Parameters: in_channels=25, initial_filters=32, num_classes=5, bilinear=False.

   * Architecture Flow:
       1. Input: (N, 25, H, W)
       2. Inc: DoubleConv(25, 32) processes the Input -> inc_output
       3. Encoder 1: Down(32, 64) processes inc_output -> encoder1_output
       4. Encoder 2: Down(64, 128) processes encoder1_output -> encoder2_output
       5. Encoder 3: Down(128, 256) processes encoder2_output -> encoder3_output
       6. Bottleneck: Down(256, 512) processes encoder3_output -> bottleneck_output
       7. Decoder 1: Up(512, 256, 256) receives bottleneck_output and encoder3_output ->
          decoder1_output
       8. Decoder 2: Up(256, 128, 128) receives decoder1_output and encoder2_output ->
          decoder2_output
       9. Decoder 3: Up(128, 64, 64) receives decoder2_output and encoder1_output ->
          decoder3_output
       10. Decoder 4: Up(64, 32, 32) receives decoder3_output and inc_output -> decoder4_output
       11. Output: OutConv(32, 5) receives decoder4_output -> final_output (shape (N, 5, H, W))
"""
import torch
import torch.nn as nn
from .base_blocks import DoubleConv, Down, Up, OutConv

class UNetBase(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True, initial_filters=64, depth=4, attention_block=None, **kwargs):
        super(UNetBase, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.depth = depth

        # --- Encoder ---
        self.inc = self.create_conv_block(in_channels, initial_filters, attention_block)
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            in_f = initial_filters * (2**i)
            out_f = in_f * 2
            self.encoders.append(
                self.create_down_block(in_f, out_f, attention_block)
            )

        # --- Bottleneck ---
        factor = 2 if bilinear else 1
        bottleneck_in = initial_filters * (2**(depth - 1))
        bottleneck_out = bottleneck_in * 2 // factor
        self.bottleneck = self.create_down_block(bottleneck_in, bottleneck_out, attention_block)


        # --- Decoder ---
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            # Số kênh từ tầng decoder trước (tầng sâu hơn)
            up_in_channels = initial_filters * (2**(i+1)) // factor
            # Số kênh từ skip connection tương ứng
            up_out_channels = initial_filters * (2**i)
            # Số kênh đầu ra của khối Up này
            out_ch = initial_filters * (2**i)
            self.decoders.append(
                Up(up_in_channels, up_out_channels, out_ch, bilinear)
            )

        self.outc = OutConv(initial_filters, num_classes)

    def create_conv_block(self, in_c, out_c, attention_block):
        if attention_block:
            return nn.Sequential(DoubleConv(in_c, out_c), attention_block(out_c))
        return DoubleConv(in_c, out_c)

    def create_down_block(self, in_c, out_c, attention_block):
        if attention_block:
            return nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_c, out_c),
                attention_block(out_c)
            )
        return Down(in_c, out_c)

    def forward(self, x):
        skip_connections = []
        
        x = self.inc(x)
        skip_connections.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        x = self.bottleneck(x) # Bottleneck is now a separate block

        # Decoder path
        skip_connections = skip_connections[::-1] # Reverse for popping
        
        for i, decoder in enumerate(self.decoders):
            # x là output từ tầng decoder trước đó hoặc bottleneck
            # skip_connections[i] là skip connection tương ứng
            x = decoder(x, skip_connections[i])

        logits = self.outc(x)
        return logits