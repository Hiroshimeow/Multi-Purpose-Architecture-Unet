import torch
import torch.nn as nn
from .unet_base import UNetBase
from .attention_modules import SpatialAttentionBlock

class DeployTABS(nn.Module):
    """
    Phiên bản Deployment của TABS.
    - Input: Tensor (B, k, H, W) -> Đã được cắt sẵn k band từ bên ngoài.
    - Không có lớp BandSelector.
    - Vẫn giữ kiến trúc UNet + SpatialAttention để tận dụng weights đã train.
    """
    def __init__(self, in_channels=5, num_classes=5, initial_filters=6, **kwargs):
        super().__init__()
        # Khởi tạo UNetBase giống hệt như cách TABS khởi tạo nó
        # Quan trọng: Phải có attention_block=SpatialAttentionBlock
        self.unet = UNetBase(
            in_channels=in_channels,
            num_classes=num_classes,
            initial_filters=initial_filters,
            attention_block=SpatialAttentionBlock, 
            **kwargs
        )

    def forward(self, x):
        # x shape: (Batch, 5, H, W)
        # Không cần bước selection nữa
        return self.unet(x)
