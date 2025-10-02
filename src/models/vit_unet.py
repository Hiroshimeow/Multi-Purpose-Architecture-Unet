import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base_blocks import DoubleConv, OutConv

# --- Helper Modules ---

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        skip_connections = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            skip_connections.append(x)
        return x, skip_connections

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + out_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- ViT UNet ---

class ViT_UNet(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 in_channels=25,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 bilinear=True,
                 skip_connection_indices = [3, 6, 9]):
        super().__init__()

        self.image_size = image_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.h = image_size // patch_size
        self.w = image_size // patch_size
        self.skip_connection_indices = skip_connection_indices

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        decoder_channels = [dim, dim // 2, dim // 4, dim // 8]
        
        # Bottleneck
        self.bottleneck_conv = DoubleConv(dim, decoder_channels[0])

        for i in range(len(decoder_channels) -1):
            self.decoders.append(
                Up(decoder_channels[i], decoder_channels[i+1], bilinear)
            )
            # Projection for skip connection
            self.skip_projections.append(nn.Linear(dim, decoder_channels[i+1]))

        self.outc = OutConv(decoder_channels[-1], num_classes)

    def forward(self, img):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer Encoder
        x, skip_connections = self.transformer(x)

        # Reshape for Decoder
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.h, self.w)

        # Decoder
        x = self.bottleneck_conv(x)
        
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[self.skip_connection_indices[i]]
            skip = self.skip_projections[i](skip)
            skip = skip.transpose(1, 2).reshape(skip.shape[0], skip.shape[2], self.h, self.w)
            x = decoder(x, skip)

        logits = self.outc(x)
        return {'segmentation': F.interpolate(logits, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)}