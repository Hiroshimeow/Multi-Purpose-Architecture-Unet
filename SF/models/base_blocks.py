# src/models/base_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
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

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels_from_prev_stage, in_channels_skip_connection, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_up = nn.Conv2d(in_channels_from_prev_stage, out_channels, kernel_size=1)
            self.conv = DoubleConv(out_channels + in_channels_skip_connection, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_from_prev_stage, in_channels_from_prev_stage // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels_from_prev_stage // 2) + in_channels_skip_connection, out_channels)

    def forward(self, x1, x2): # x1 is input from previous decoder stage, x2 is skip connection
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- Blocks for FlashUNet (Depthwise Separable) ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleDSConv(nn.Module):
    """(Depthwise Separable Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_ds_conv(x)

# --- LearnableBandSelector (Imported from TABS project) ---
class LearnableBandSelector(nn.Module):
    """
    Selects the top-k bands from an input tensor using a learnable,
    differentiable Gumbel-Top-k mechanism.
    """
    def __init__(self, in_channels: int, num_bands_to_select: int):
        super().__init__()
        if num_bands_to_select > in_channels:
            raise ValueError("num_bands_to_select cannot be greater than in_channels.")
        self.in_channels = in_channels
        self.k = num_bands_to_select
        # Learnable logits for each band
        self.logits = nn.Parameter(torch.randn(in_channels))
        # Temperature for Gumbel-Softmax (can be annealed)
        self.temperature = 1.0

    def _gumbel_topk_sampling(self, logits, k, temperature=1.0):
        """
        Differentiable Top-k sampling using the Gumbel-Softmax trick with Straight-Through Estimator (STE).
        """
        # 1. Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, device=logits.device) + 1e-20) + 1e-20)
        
        # 2. Add noise to logits and apply softmax (Soft approximation)
        # Use provided temperature
        y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        # 3. Generate Hard Mask (Discrete) using Top-k
        _, top_k_indices = torch.topk(logits + gumbel_noise, k=k, dim=-1)
        
        # Create hard one-hot mask
        y_hard = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
        
        # 4. Straight-Through Estimator (STE)
        # y = y_soft + (y_hard - y_soft).detach()
        y = y_soft + (y_hard - y_soft).detach()
        
        return y, top_k_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        if C != self.in_channels:
            raise ValueError(f"Input tensor has {C} channels, but selector was initialized for {self.in_channels}.")

        # Get the selection mask with STE
        selection_mask, top_k_indices = self._gumbel_topk_sampling(self.logits, self.k, self.temperature)
        
        # Expand mask for broadcasting: (C,) -> (1, C, 1, 1)
        mask_expanded = selection_mask.view(1, C, 1, 1)
        
        # Apply mask to input.
        x_masked = x * mask_expanded
        
        # Select the top-k channels.
        dense_output = torch.index_select(x_masked, dim=1, index=top_k_indices)

        return dense_output
