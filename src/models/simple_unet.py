# src/models/simple_unet.py
import torch.nn as nn
from .unet_base import UNetBase

class SimpleUNet(nn.Module):
    """
    A simple, configurable U-Net model that acts as a direct wrapper around UNetBase.
    This is intended for straightforward segmentation tasks where no special
    preprocessing or channel manipulation is needed at the model's entry point.
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        self.unet = UNetBase(
            in_channels=in_channels, 
            num_classes=num_classes,
            **kwargs
        )

    def forward(self, x):
        return self.unet(x)
