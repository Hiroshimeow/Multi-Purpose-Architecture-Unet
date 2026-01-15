"""
Implementation of the CB-SFNet using the standardized U-Net backbone for fair comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import StandardDoubleConv, StandardDown, StandardUp, StandardOutConv
from .cbsfnet import CSBD, MDSA_Net # Reuse the novel modules

# ============================================================
# Unified CB-SFNet Architecture
# ============================================================

class CB_SFNet_Unified(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_filters: int = 64):
        super(CB_SFNet_Unified, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # --- Novelty Modules (Reused from original CB-SFNet) ---
        self.csbd = CSBD(in_channels=in_channels, out_channels=1)
        mdsa_out_channels = base_filters * 4 # As defined in the original CB-SFNet
        self.mdsa_net = MDSA_Net(in_channels=in_channels, boundary_channels=1, out_channels=mdsa_out_channels)

        # --- Standardized U-Net Backbone Parts ---
        self.inc = StandardDoubleConv(in_channels, base_filters)
        self.down1 = StandardDown(base_filters, base_filters * 2)
        self.down2 = StandardDown(base_filters * 2, base_filters * 4)
        self.down3 = StandardDown(base_filters * 4, base_filters * 8)
        
        # Bottleneck convolution now takes the original encoder output + MDSA features
        self.bottleneck_conv = StandardDoubleConv(base_filters * 8 + mdsa_out_channels, base_filters * 16)

        self.up1 = StandardUp(base_filters * 16, base_filters * 8)
        self.up2 = StandardUp(base_filters * 8, base_filters * 4)
        self.up3 = StandardUp(base_filters * 4, base_filters * 2)
        self.up4 = StandardUp(base_filters * 2, base_filters)
        self.outc = StandardOutConv(base_filters, num_classes)

    def forward(self, x: torch.Tensor):
        # 1. Contextual Spectral Boundary Discovery
        boundary_logits = self.csbd(x)
        boundary_features = torch.sigmoid(boundary_logits)

        # 2. U-Net Encoder Path (4 levels)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 3. Multi-Depth Semantic Aggregation
        x_bottleneck_in = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        boundary_bottleneck = F.interpolate(boundary_features, size=x4.shape[2:], mode='bilinear', align_corners=False)
        mdsa_features = self.mdsa_net(x_bottleneck_in, boundary_bottleneck)

        # 4. Inject MDSA features at the bottleneck
        x4_enhanced = torch.cat([x4, mdsa_features], dim=1)
        x5 = self.bottleneck_conv(x4_enhanced)

        # 5. U-Net Decoder Path
        # The skip connection for the first up-sampling step is the original x4 from the encoder
        x_up = self.up1(x5, x4) 
        x_up = self.up2(x_up, x3)
        x_up = self.up3(x_up, x2)
        x_up = self.up4(x_up, x1)

        # 6. Final Segmentation Output
        seg_logits = self.outc(x_up)

        # 7. Return dictionary for multi-task loss
        return {
            'segmentation': seg_logits,
            'boundary': boundary_logits,
            'contrast_features': x_up
        }