# src/models/attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- LearnableBandSelector (using Gumbel-Softmax) ---
class LearnableBandSelector(nn.Module):
    """
    Selects the top-k bands from an input tensor using a learnable,
    differentiable Gumbel-Top-k mechanism.
    This is the new 'BandSelector' for the project.
    """
    def __init__(self, in_channels: int, num_bands_to_select: int):
        super().__init__()
        if num_bands_to_select > in_channels:
            raise ValueError("num_bands_to_select cannot be greater than in_channels.")
        self.in_channels = in_channels
        self.k = num_bands_to_select
        # Learnable logits for each band
        self.logits = nn.Parameter(torch.randn(in_channels))

    def _gumbel_topk_sampling(self, logits, k, temperature=1.0):
        """
        Differentiable Top-k sampling using the Gumbel-Softmax trick with Straight-Through Estimator (STE).
        """
        # 1. Sample Gumbel noise
        # g = -log(-log(u))
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, device=logits.device) + 1e-20) + 1e-20)
        
        # 2. Add noise to logits and apply softmax (Soft approximation)
        y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        # 3. Generate Hard Mask (Discrete) using Top-k
        # We use the perturbed logits to find indices, preserving stochasticity
        _, top_k_indices = torch.topk(logits + gumbel_noise, k=k, dim=-1)
        
        # Create hard one-hot mask
        y_hard = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
        
        # 4. Straight-Through Estimator (STE)
        # Forward pass uses y_hard (binary). Backward pass uses y_soft (gradients).
        # y = y_soft + (y_hard - y_soft).detach()
        # This allows gradients to flow from the output back to 'logits' via 'y_soft'
        y = y_soft + (y_hard - y_soft).detach()
        
        return y, top_k_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        if C != self.in_channels:
            raise ValueError(f"Input tensor has {C} channels, but selector was initialized for {self.in_channels}.")

        # Get the selection mask with STE
        # selection_mask shape: (C,) or (B, C) depending on implementation. 
        # Here logits are (C,), so mask is (C,)
        selection_mask, top_k_indices = self._gumbel_topk_sampling(self.logits, self.k)
        
        # Expand mask for broadcasting: (C,) -> (1, C, 1, 1)
        mask_expanded = selection_mask.view(1, C, 1, 1)
        
        # Apply mask to input.
        # Crucial: This multiplication connects x and mask in the computation graph.
        # Since 'mask' contains 'y_soft' in its backward path, gradients will flow to logits.
        x_masked = x * mask_expanded
        
        # Select the top-k channels.
        # We rely on the fact that non-selected channels are zeroed out (or close to it in soft version).
        # However, to physically reduce channels to 'k' for the UNet, we must index.
        # Even though 'index_select' is technically not differentiable w.r.t indices,
        # the values at those indices come from 'x_masked', which IS differentiable w.r.t logits.
        
        # Note: If logits are shared across batch (default), we just take the indices from the single logit vector.
        # If we wanted per-sample selection, we'd need more complex gathering.
        # Here logits is (C,), so top_k_indices is (k,).
        
        dense_output = torch.index_select(x_masked, dim=1, index=top_k_indices)

        return dense_output

# For backward compatibility, BandSelector is now an alias for LearnableBandSelector
BandSelector = LearnableBandSelector


# --- SAC (Spectral Attention Conv) with internal SE_attention ---
class SE_attention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_attention, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=[3, 4])
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        return x * y

class SAC(nn.Module):
    def __init__(self, input_depth=25, reduced_depth=3, channels=32, use_sea=True):
        super(SAC, self).__init__()
        self.use_sea = use_sea
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        if self.use_sea:
            self.attention = SE_attention(channels, reduction=4)
        
        kernel_depth = input_depth - reduced_depth + 1
        self.reduce_conv = nn.Conv3d(channels, 1, kernel_size=(kernel_depth,1,1))
        
        self.skip_conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        self.down_conv = nn.Conv3d(1, 1, kernel_size=(kernel_depth,1,1))

    def forward(self, x):
        x_in = x.unsqueeze(1)
        x_branch = self.conv1(x_in)
        x_branch = self.conv2(x_branch)
        if self.use_sea:
            x_branch = self.attention(x_branch)
        x_branch = self.reduce_conv(x_branch)
        
        x_skip = self.skip_conv(x_in)
        x_skip = self.down_conv(x_skip)
        
        out = F.relu(x_branch + x_skip)
        return out.squeeze(1)

# --- CBAM (Convolutional Block Attention Module) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SpatialAttentionBlock(nn.Module):
    """Wrapper to apply SpatialAttention and multiply by input."""
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return x * self.sa(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- BCD (Band Compression Decompression) ---
class BandCompressionDecompression(nn.Module):
    def __init__(self, in_channels, intermediate_channels):
        super(BandCompressionDecompression, self).__init__()
        self.compress = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1)
        self.decompress = nn.Conv2d(intermediate_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.compress(x))
        x = self.decompress(x)
        return x

# --- Attention Gate (AG) for U-Net Skip Connections ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False))
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi