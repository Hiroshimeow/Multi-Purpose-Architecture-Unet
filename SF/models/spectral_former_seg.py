import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from .base_blocks import DoubleConv, OutConv, LearnableBandSelector

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CAFTransformer(nn.Module):
    """Transformer with Cross-layer Adaptive Fusion (CAF)"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, num_patches):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        
        # CAF specific: skip connections across layers
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 1):
            # Fusing current layer output with previous layer output
            # Fixed kernel size type error: [1, 2] -> (1, 2)
            self.skipcat.append(nn.Conv2d(num_patches, num_patches, (1, 2), 1, 0))

    def forward(self, x):
        outputs = []
        # Fixed iteration over ModuleList: layers is iterable
        for i, block in enumerate(self.layers):
            attn, ff = block[0], block[1]
            x = attn(x)
            x = ff(x)
            
            if i > 0:
                # CAF: fuse current x with previous output
                # x: [B, N, D], outputs[-1]: [B, N, D]
                # Combine along a new dimension for Conv2d: [B, N, D, 2]
                fused = torch.stack([x, outputs[-1]], dim=-1) # [B, N, D, 2]
                x = self.skipcat[i-1](fused).squeeze(-1)
            
            outputs.append(x)
        return x, outputs

class SpectralFormer_Seg(nn.Module):
    def __init__(self, 
                 image_size=224, 
                 in_channels=25, 
                 num_classes=5, 
                 patch_size=16,
                 dim=128, 
                 depth=6, 
                 heads=8, 
                 mlp_dim=256, 
                 dim_head=16, 
                 dropout=0.1, 
                 emb_dropout=0.1,
                 num_selected_bands=None): # Added argument
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        # --- Band Selection Logic ---
        self.band_selector = None
        input_dim_for_transformer = in_channels

        if num_selected_bands is not None and num_selected_bands < in_channels:
             print(f"SpectralFormer: Initializing LearnableBandSelector (k={num_selected_bands}/{in_channels})")
             self.band_selector = LearnableBandSelector(in_channels, num_selected_bands)
             input_dim_for_transformer = num_selected_bands
        
        patch_dim = input_dim_for_transformer * patch_size ** 2
        
        self.h = image_size // patch_size
        self.w = image_size // patch_size

        # GSE could be implemented as a specialized patch embedding or a grouping layer
        # Here we use a standard linear embedding but the architecture allows CAF
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = CAFTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)

        # --- Dynamic Decoder ---
        # Calculate how many upsampling steps are needed: log2(patch_size)
        num_upsamples = int(np.log2(patch_size))
        decoder_layers = []
        curr_dim = dim
        
        # Intermediate channel progression
        channels = [max(16, dim // (2**i)) for i in range(num_upsamples)]
        
        for i in range(num_upsamples):
            out_dim = channels[i]
            decoder_layers.append(DoubleConv(curr_dim, out_dim))
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            curr_dim = out_dim
            
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_conv = OutConv(curr_dim, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        
        # Apply Band Selection if active
        if self.band_selector is not None:
             x = self.band_selector(x)
             # Output is [B, k, H, W]
        
        p = self.patch_size
        
        # Patch embedding
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer Encoder (CAF)
        x, _ = self.transformer(x)

        # Reshape for Decoder: [B, N, D] -> [B, D, H/p, W/p]
        x = rearrange(x, 'b (h w) d -> b d h w', h=self.h, w=self.w)

        # Decoder
        x = self.decoder(x)
        logits = self.final_conv(x)
        
        return {'segmentation': logits}
