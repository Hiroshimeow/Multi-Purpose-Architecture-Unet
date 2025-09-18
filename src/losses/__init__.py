# src/losses/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the new Gemini loss
from .loss import GeminiCombinedLoss

# --- Loss classes from losses.py ---

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        targets = targets.long()
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class FocalLoss(nn.Module):
    """
    [NEW] Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = torch.FloatTensor(weight) if weight is not None else None
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device
        # Use reduction='none' to get per-element loss
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none', weight=self.weight.to(device) if self.weight is not None else None)
        # pt is the probability of the correct class
        pt = torch.exp(-ce_loss)
        # This is the core focal loss formula
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedCombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=1.0, beta=0.5):
        super(WeightedCombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weight) if weight is not None else None,
            reduction='none'
        )
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, weight_map):
        ce_pixel_loss = self.ce_loss(logits, targets.long())
        weighted_ce_loss = (ce_pixel_loss * weight_map).mean()
        dice = self.dice_loss(logits, targets)
        return self.alpha * weighted_ce_loss + self.beta * dice

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5, beta=0.5, ds_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight) if weight is not None else None)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta
        self.ds_weights = ds_weights if ds_weights else [0.4, 0.2] # Default weights for DS outputs

    def forward(self, outputs, targets):
        # Handle both single tensor output and list output (for deep supervision)
        if not isinstance(outputs, list):
            outputs = [outputs]

        total_loss = 0
        # The first output is the main one, gets full weight
        main_loss = self.alpha * self.ce_loss(outputs[0], targets.long()) + self.beta * self.dice_loss(outputs[0], targets)
        total_loss += main_loss

        # Add weighted losses for deep supervision outputs
        if len(outputs) > 1:
            for i, (output, weight) in enumerate(zip(outputs[1:], self.ds_weights)):
                # Downsample target mask to match output size
                target_downsampled = F.interpolate(targets.unsqueeze(1).float(), size=output.shape[2:], mode='nearest').squeeze(1)
                ds_loss = self.alpha * self.ce_loss(output, target_downsampled.long()) + self.beta * self.dice_loss(output, target_downsampled)
                total_loss += weight * ds_loss
        
        return total_loss

# --- Loss Factory ---

def get_loss(name: str, params: dict, class_weights: np.ndarray, device):
    """
    Loss function factory.
    """
    print(f"   Initializing loss: {name}")

    if name == 'GeminiCombinedLoss':
        return GeminiCombinedLoss(class_weights=class_weights, device=device, **params)
    
    # For other losses, convert numpy array to list
    if class_weights is not None:
        params['weight'] = class_weights.tolist()

    if name == 'WeightedCombinedLoss':
        return WeightedCombinedLoss(**params)
    elif name == 'CombinedLoss':
        return CombinedLoss(**params)
    elif name == 'FocalLoss':
        return FocalLoss(**params)
    else:
        raise ValueError(f"Loss function '{name}' not recognized.")
