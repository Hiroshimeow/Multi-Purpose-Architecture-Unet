# src/models/unet_prunable.py
import torch.nn as nn
from .unet_base import UNetBase
from .pruning_modules import ChannelWeighter

class UnetPrunable(nn.Module):
    """
    A U-Net variant designed for channel pruning experiments.

    This model incorporates a ChannelWeighter module at the beginning to learn
    the importance of each input channel. It is trained on all input channels,
    and the weights from the ChannelWeighter are used with a regularization
    loss (e.g., L1) to encourage sparsity.
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        if 'attention_block' in kwargs:
            print("Warning: 'attention_block' is not used in UnetPrunable and will be ignored.")
            kwargs.pop('attention_block', None)

        self.weighter = ChannelWeighter(in_channels=in_channels)
        
        # The UNetBase processes the full number of channels after re-weighting
        self.unet = UNetBase(
            in_channels=in_channels, 
            num_classes=num_classes,
            attention_block=None, # No other attention applied
            **kwargs
        )

    def forward(self, x):
        """
        Forward pass for the prunable U-Net.

        Returns a dictionary containing the segmentation output and the channel weights
        for use in a specialized pruning loss function.
        """
        # Apply learnable weights to channels
        weighted_x = self.weighter(x)
        
        # Get segmentation output from the U-Net
        segmentation_output = self.unet(weighted_x)
        
        # Return a dictionary for the multi-component loss function
        return {
            'segmentation': segmentation_output,
            'channel_weights': self.weighter.channel_weights
        }
