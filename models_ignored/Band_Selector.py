import torch
import torch.nn as nn

class BandSelector(nn.Module):
    def __init__(self, original_in_channels, num_selected_bands):
        super().__init__()
        self.original_in_channels = original_in_channels
        self.num_selected_bands = num_selected_bands

        # Learn a transformation matrix that maps original channels to selected bands
        # Each row of this matrix will be a set of weights for the original channels
        # that sum up to form one of the 'selected' bands.
        # Initialized with small random values to encourage learning.
        self.selection_weights = nn.Parameter(torch.randn(num_selected_bands, original_in_channels) * 0.01)

    def forward(self, x):
        # x shape: (B, C_orig, H, W) where C_orig is original_in_channels (25)

        # Apply softmax across the original channels for each output band
        # This ensures that for each 'selected' band, the sum of weights from original bands is 1.
        # Shape of softmax_weights: (num_selected_bands, original_in_channels)
        softmax_weights = torch.softmax(self.selection_weights, dim=1)

        # Perform a weighted sum using einsum
        # 's' for selected bands, 'c' for original channels, 'b' for batch, 'h' for height, 'w' for width
        # Resulting shape: (B, num_selected_bands, H, W)
        selected_bands = torch.einsum('sc,bchw->bshw', softmax_weights, x)

        return selected_bands
