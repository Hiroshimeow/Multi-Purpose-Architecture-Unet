
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Copied from src/models/base_blocks.py for self-containment ---

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    """ This is the 'older' Up block that unet_base_1.py seems to be incompatible with.
        This version takes 3 channel arguments. Let's see if the call in unet_base_1 matches it.
    """
    def __init__(self, in_channels_from_prev_stage, in_channels_skip_connection, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # This conv_up was not in the very first Up version I saw. This seems to be the source of incompatibility.
            # This implementation IS DIFFERENT from the one implied by unet_base.py (current).
            self.conv = DoubleConv(in_channels_from_prev_stage + in_channels_skip_connection, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_from_prev_stage, in_channels_from_prev_stage // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels_from_prev_stage // 2) + in_channels_skip_connection, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

# --- Copied and renamed class from src/models/unet_base_1.py ---

class UNetBaseV1(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True, initial_filters=64, depth=4, attention_block=None, **kwargs):
        super(UNetBaseV1, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.depth = depth
        self.factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, initial_filters)
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            in_f = initial_filters * (2**i)
            out_f = in_f * 2
            self.encoders.append(Down(in_f, out_f))

        bottleneck_in = initial_filters * (2**(depth - 1))
        bottleneck_out = bottleneck_in * 2 // self.factor
        self.bottleneck = Down(bottleneck_in, bottleneck_out)

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_in_channels = initial_filters * (2**(i+1)) // self.factor
            up_out_channels = initial_filters * (2**i)
            # The 'Up' block this was designed for likely had a different signature
            # The current 'Up' block expects 3 channel arguments. Let's try to adapt.
            # Up(in_prev, in_skip, out)
            # The input to DoubleConv in the working Up block is (in_prev // 2) + in_skip
            # Let's assume the call was meant for a simpler Up block.
            # To make this work, we need to understand the old Up block's __init__
            # Based on the original code I wrote, it was likely Up(in_channels, out_channels, bilinear)
            # where in_channels = up_in + skip_conn.
            # Let's try to call the current Up block with the right arguments, which seems to have been the intent.
            # It seems `unet_base_1.py` call was `Up(up_in_channels, up_out_channels, out_ch, bilinear)`
            # The issue is not the call, but which 'Up' version is being used.
            # Let's redefine the Up class here to be what `unet_base.py` (current) uses.
            
            # Re-defining a compatible Up block locally for this test
            _up_in = initial_filters * (2**(i+1)) // self.factor
            _skip_in = initial_filters * (2**i)
            _out = initial_filters * (2**i) // self.factor
            self.decoders.append(
                Up(_up_in, _skip_in, _out, bilinear)
            )

        self.outc = OutConv(initial_filters, num_classes) # The key difference is here - no "// factor"

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

# --- Calculation Logic ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("--- Calculating Parameter Count for UNetBaseV1 (from unet_base_1.py) ---")
try:
    # Target configuration
    depth = 4
    initial_filters = 6
    
    # Common parameters for instantiation
    in_channels = 3 
    num_classes = 5

    model = UNetBaseV1(
        in_channels=in_channels, 
        num_classes=num_classes, 
        initial_filters=initial_filters, 
        depth=depth,
        bilinear=True
    )
    params_m = count_parameters(model) / 1_000_000
    print(f"\nConfiguration: d={depth}, i={initial_filters}")
    print(f"Parameter Count: {params_m:.2f}M")

except Exception as e:
    import traceback
    print(f"\nAn error occurred: {e}")
    traceback.print_exc()

# Redefine the Up block from the *current* base_blocks.py to ensure compatibility
class Up(nn.Module):
    def __init__(self, in_channels_from_prev_stage, in_channels_skip_connection, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # The conv_up is a key part that might have been missing in an older version.
            self.conv_up = nn.Conv2d(in_channels_from_prev_stage, out_channels, kernel_size=1)
            self.conv = DoubleConv(out_channels + in_channels_skip_connection, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_from_prev_stage, in_channels_from_prev_stage // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels_from_prev_stage // 2) + in_channels_skip_connection, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
