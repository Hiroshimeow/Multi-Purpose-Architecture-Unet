"""
Full implementation of the Contextual Boundary-Aware Spectral Fusion Network (CB-SFNet)
with Attention Gates (AGs) in the decoder skip connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_modules import AttentionGate # Import the new module


# ============================================================
# Sub-Module Implementations (Keeping them as they are)
# ============================================================

class SpectralGradientAnalyzer(nn.Module):
    """Computes gradients between adjacent spectral bands."""
    def __init__(self):
        super(SpectralGradientAnalyzer, self).__init__()
        # Using a fixed 1D conv to compute differences between adjacent channels
        # Kernel weights are [-1, 1], applied channel-wise
        self.filter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
        with torch.no_grad():
            self.filter.weight.copy_(torch.tensor([[[-1.0, 1.0]]], dtype=torch.float32))
        for p in self.filter.parameters():
            p.requires_grad = False  # fixed operation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        if C < 2:
            # Cannot compute gradient with less than 2 channels, return zero tensor
            return torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        # Reshape to apply 1D conv across the channel dimension
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)

        # Compute gradients
        grad = self.filter(x_reshaped)

        # Take absolute value and reshape back
        grad = torch.abs(grad)
        grad = grad.reshape(B, H, W, C - 1).permute(0, 3, 1, 2)

        # Sum the gradients to get a single-channel boundary representation
        boundary_map = torch.sum(grad, dim=1, keepdim=True)
        return boundary_map


class CSBD(nn.Module):
    """Contextual Spectral Boundary Discovery - Full Implementation"""
    def __init__(self, in_channels: int, mid_channels: int = 16, out_channels: int = 1):
        super(CSBD, self).__init__()
        self.spectral_gradient = SpectralGradientAnalyzer()
        # Boundary Localization Network (a small CNN)
        self.boundary_cnn = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)  # Outputs boundary logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectral_grads = self.spectral_gradient(x)
        boundary_logits = self.boundary_cnn(spectral_grads)
        return boundary_logits


class MDSA_Net(nn.Module):
    """
    Multi-Depth Semantic Aggregation Network - Adapted for 3-channel input.
    The spirit of spectral stratification is maintained via different processing paths.
    """
    def __init__(self, in_channels: int, boundary_channels: int, out_channels: int):
        super(MDSA_Net, self).__init__()
        # Path 1: Surface Material (Lightweight Conv)
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels + boundary_channels, out_channels // 4, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        # Path 2: Structural Pattern (Multi-scale Pyramid)
        self.path2_conv1 = nn.Conv2d(in_channels + boundary_channels, out_channels // 4, kernel_size=1)
        self.path2_pool1 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.path2_pool2 = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)

        # Path 3: Complex Mixture (Residual Block)
        res_channels = out_channels - (out_channels // 4) * 2
        self.path3_conv1 = nn.Conv2d(in_channels + boundary_channels, res_channels, kernel_size=1)
        self.path3_resblock = nn.Sequential(
            nn.Conv2d(res_channels, res_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, res_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(res_channels)
        )
        self.path3_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, boundary_features: torch.Tensor) -> torch.Tensor:
        # Cross-Level Boundary Enhancement (via concatenation)
        x_enhanced = torch.cat([x, boundary_features], dim=1)

        # Path 1
        out1 = self.path1(x_enhanced)

        # Path 2
        p2_base = self.path2_conv1(x_enhanced)
        p2_p1 = self.path2_pool1(p2_base)
        p2_p2 = self.path2_pool2(p2_base)
        out2 = p2_base + p2_p1 + p2_p2

        # Path 3
        p3_base = self.path3_conv1(x_enhanced)
        p3_res = self.path3_resblock(p3_base)
        out3 = self.path3_relu(p3_base + p3_res)

        # Aggregate features
        out = torch.cat([out1, out2, out3], dim=1)
        return out


# ============================================================
# U-Net Building Blocks
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# Main CB-SFNet Architecture with Attention Gates
# ============================================================

class CB_SFNet_AG(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_filters: int = 64, bilinear: bool = False):
        super(CB_SFNet_AG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # --- Novelty Modules ---
        self.csbd = CSBD(in_channels=in_channels, out_channels=1)
        mdsa_out_channels = base_filters * 4
        self.mdsa_net = MDSA_Net(in_channels=in_channels, boundary_channels=1, out_channels=mdsa_out_channels)

        # --- Base U-Net Backbone ---
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8 + mdsa_out_channels, base_filters * 16 // factor)

        # --- Attention Gates ---
        self.ag1 = AttentionGate(F_g=base_filters * 16 // factor, F_l=base_filters * 8, F_int=base_filters * 4)
        self.ag2 = AttentionGate(F_g=base_filters * 8 // factor, F_l=base_filters * 4, F_int=base_filters * 2)
        self.ag3 = AttentionGate(F_g=base_filters * 4 // factor, F_l=base_filters * 2, F_int=base_filters)
        self.ag4 = AttentionGate(F_g=base_filters * 2 // factor, F_l=base_filters, F_int=base_filters // 2)

        # --- Decoder ---
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        self.outc = OutConv(base_filters, num_classes)

    def forward(self, x: torch.Tensor):
        # 1. Contextual Spectral Boundary Discovery
        boundary_logits = self.csbd(x)
        boundary_features = torch.sigmoid(boundary_logits)

        # 2. U-Net Encoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4_pre_mdsa = self.down3(x3)

        # 3. Multi-Depth Semantic Aggregation
        x_bottleneck = F.interpolate(x, size=x4_pre_mdsa.shape[2:], mode='bilinear', align_corners=False)
        boundary_bottleneck = F.interpolate(boundary_features, size=x4_pre_mdsa.shape[2:], mode='bilinear', align_corners=False)
        mdsa_features = self.mdsa_net(x_bottleneck, boundary_bottleneck)

        # 4. Integrate MDSA features at the bottleneck
        x4_enhanced = torch.cat([x4_pre_mdsa, mdsa_features], dim=1)
        x5 = self.down4(x4_enhanced)

        # 5. U-Net Decoder Path with Attention Gates
        d5 = self.up1(x5, self.ag1(g=x5, x=x4_pre_mdsa))
        d4 = self.up2(d5, self.ag2(g=d5, x=x3))
        d3 = self.up3(d4, self.ag3(g=d4, x=x2))
        d2 = self.up4(d3, self.ag4(g=d3, x=x1))
        
        # 6. Final Segmentation Output
        seg_logits = self.outc(d2)

        # Return a dictionary of outputs for the multi-task loss
        return {
            'segmentation': seg_logits,
            'boundary': boundary_logits,
            'contrast_features': d2
        }
