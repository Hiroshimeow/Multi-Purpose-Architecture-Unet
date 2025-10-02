import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Compact U-Net Components (Stage 3) ---

class AdaptiveSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        self.context_pool = nn.AdaptiveAvgPool2d(1)
        self.context_fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        spatial_att = self.spatial_attention(x)
        global_context = self.context_pool(x).view(B, C)
        channel_att = self.context_fc(global_context).view(B, C, 1, 1)
        attended_x = x * spatial_att * channel_att
        return attended_x

class CompactEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_asa=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_attention = AdaptiveSpatialAttention(out_channels) if use_asa else None
    
    def forward(self, x):
        x = self.conv_block(x)
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        return x

class CompactEncoder(nn.Module):
    def __init__(self, in_channels, base_filters, stages):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(stages):
            out_ch = base_filters * (2 ** i)
            self.encoder_blocks.append(
                CompactEncoderBlock(in_channels=in_ch, out_channels=out_ch, use_asa=(i >= 2))
            )
            in_ch = out_ch
    
    def forward(self, x):
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
        return x, skip_connections

class CompactDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.up(x)
        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)

class CompactDecoder(nn.Module):
    def __init__(self, base_filters, stages):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        for i in range(stages - 1, -1, -1):
            in_ch = base_filters * (2 ** (i + 1))
            skip_ch = base_filters * (2 ** i)
            out_ch = base_filters * (2 ** i)
            self.decoder_blocks.append(
                CompactDecoderBlock(in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch)
            )

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skip_connections[i])
        return x

class CompactUNetWithASA(nn.Module):
    def __init__(self, in_channels=25, num_classes=19, base_filters=64):
        super().__init__()
        self.stages = 4
        self.encoder = CompactEncoder(in_channels=in_channels, base_filters=base_filters, stages=self.stages)
        self.bottleneck = CompactEncoderBlock(
            in_channels=base_filters * (2 ** (self.stages - 1)),
            out_channels=base_filters * (2 ** self.stages),
            use_asa=True
        )
        self.decoder = CompactDecoder(base_filters=base_filters, stages=self.stages)
        self.classifier = nn.Conv2d(base_filters, num_classes, 1)
    
    def forward(self, enhanced_bands):
        bottleneck_input, skip_connections = self.encoder(enhanced_bands)
        x = self.bottleneck(bottleneck_input)
        decoder_output = self.decoder(x, skip_connections)
        return self.classifier(decoder_output)

# --- Spectral Reconstruction Components (Stage 1) ---

class SpectralContinuityAttention(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.smoothness_conv = nn.Conv1d(bands, bands, kernel_size=3, padding=1, groups=bands, bias=False)
        nn.init.constant_(self.smoothness_conv.weight, 1/3.0)

    def forward(self, spectral_bands):
        B, C, H, W = spectral_bands.shape
        spectral_vectors = spectral_bands.permute(0, 2, 3, 1).reshape(-1, C).unsqueeze(-1)
        refined_spectrum = self.smoothness_conv(spectral_vectors)
        return refined_spectrum.squeeze(-1).reshape(B, H, W, C).permute(0, 3, 1, 2)

class IntelligentSpectralReconstructor(nn.Module):
    def __init__(self, in_channels=3, output_channels=25, **kwargs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, 1),
            nn.Sigmoid()
        )
        self.continuity_enforcer = SpectralContinuityAttention(bands=output_channels)

    def forward(self, x):
        reconstructed = self.network(x)
        return self.continuity_enforcer(reconstructed)

# --- Cross-Band Attention Components (Stage 2) ---

class AdaptiveBandWeighter(nn.Module):
    def __init__(self, channels, context_size=7):
        super().__init__()
        self.context_analyzer = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels),
            nn.Sigmoid()
        )
    
    def forward(self, spectral_tokens):
        adaptive_weights = self.context_analyzer(spectral_tokens)
        return spectral_tokens * adaptive_weights

class AdaptiveCrossBandAttention(nn.Module):
    def __init__(self, channels=25, attention_heads=5, adaptive_weighting=True):
        super().__init__()
        assert channels % attention_heads == 0, "channels must be divisible by attention_heads"
        self.cross_band_attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=attention_heads, dropout=0.1, batch_first=True
        )
        self.adaptive_weighter = AdaptiveBandWeighter(channels) if adaptive_weighting else None
        self.band_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, reconstructed_bands):
        B, C, H, W = reconstructed_bands.shape
        downsampled_bands = F.adaptive_avg_pool2d(reconstructed_bands, (H // 8, W // 8))
        spectral_tokens = downsampled_bands.permute(0, 2, 3, 1).reshape(B, (H // 8) * (W // 8), C)
        attended_tokens, _ = self.cross_band_attention(spectral_tokens, spectral_tokens, spectral_tokens)
        if self.adaptive_weighter is not None:
            attended_tokens = self.adaptive_weighter(attended_tokens)
        attended_bands_small = attended_tokens.reshape(B, H // 8, W // 8, C).permute(0, 3, 1, 2)
        attended_bands = F.interpolate(attended_bands_small, size=(H, W), mode='bilinear', align_corners=False)
        fused_bands = self.band_fusion(attended_bands)
        return fused_bands + reconstructed_bands

# --- Main ASRAN Model ---

class ASRAN(nn.Module):
    def __init__(self, in_channels=3, reconstructed_channels=25, num_classes=5, base_filters=64, **kwargs):
        super().__init__()
        self.spectral_reconstructor = IntelligentSpectralReconstructor(
            in_channels=in_channels, output_channels=reconstructed_channels
        )
        self.cross_band_attention = AdaptiveCrossBandAttention(
            channels=reconstructed_channels, attention_heads=5
        )
        self.segmentation_backbone = CompactUNetWithASA(
            in_channels=reconstructed_channels, num_classes=num_classes, base_filters=base_filters
        )

    def forward(self, x):
        reconstructed_hsi = self.spectral_reconstructor(x)
        enhanced_hsi = self.cross_band_attention(reconstructed_hsi)
        segmentation_output = self.segmentation_backbone(enhanced_hsi)
        return segmentation_output