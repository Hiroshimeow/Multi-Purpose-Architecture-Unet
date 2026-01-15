import torch
import torch.nn as nn
from .base_blocks import DoubleConv, Down, Up, OutConv

class UNetBase(nn.Module):
    """
    This is the lightweight, configurable UNet that was in use before
    the attempt to restore the hardcoded version. This version's size is
    controlled by `initial_filters` and `depth`.
    """
    def __init__(self, in_channels, num_classes, bilinear=True, initial_filters=6, depth=4, attention_block=None, **kwargs):
        super(UNetBase, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.depth = depth

        factor = 2 if bilinear else 1

        # --- Encoder ---
        self.inc = DoubleConv(in_channels, initial_filters)
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            in_f = initial_filters * (2**i)
            out_f = in_f * 2
            self.encoders.append(Down(in_f, out_f))
        
        # --- Bottleneck ---
        bottleneck_in = initial_filters * (2**(depth - 1))
        self.bottleneck = Down(bottleneck_in, bottleneck_in * 2 // factor)

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_in_from_prev = initial_filters * (2**(i+1)) // factor
            skip_ch = initial_filters * (2**i)
            out_ch = initial_filters * (2**i) // factor
            
            # The logic for Up block inputs must be precise
            if i == depth - 1: # First decoder block
                 up_in_from_prev = initial_filters * (2**depth) // factor
            
            # This logic was buggy before, now corrected
            # The input to an Up block is the output of the previous Up block
            if i < depth - 1:
                 up_in_from_prev = initial_filters * (2**(i+1)) // factor

            self.decoders.append(
                Up(up_in_from_prev, skip_ch, out_ch, bilinear)
            )

        self.outc = OutConv(initial_filters // factor, num_classes)

    def forward(self, x):
        skip_connections = []
        
        x = self.inc(x)
        skip_connections.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        logits = self.outc(x)
        return logits
