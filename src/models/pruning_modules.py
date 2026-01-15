# src/models/pruning_modules.py
import torch
import torch.nn as nn

class ChannelWeighter(nn.Module):
    """
    A module that assigns a learnable weight to each input channel.
    These weights can then be used for regularization (e.g., L1) to encourage sparsity,
    effectively learning the importance of each channel.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Initialize weights close to 1
        self.channel_weights = nn.Parameter(torch.ones(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the learned channel weights to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with channels re-weighted, same shape as input.
        """
        # Ensure weights are broadcastable to (1, C, 1, 1)
        return x * self.channel_weights.view(1, -1, 1, 1)
