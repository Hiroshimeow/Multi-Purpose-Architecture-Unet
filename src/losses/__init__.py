# src/losses/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import loss classes from their respective files
from .loss import GeminiCombinedLoss, PruningLoss
from .cbsfnet_loss import CBSFNetLoss
from .base_losses import DiceLoss, CombinedLoss

# --- Loss classes defined directly in this module ---

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = torch.FloatTensor(weight) if weight is not None else None
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none', weight=self.weight.to(device) if self.weight is not None else None)
        pt = torch.exp(-ce_loss)
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
        self.dice_loss = DiceLoss() # DiceLoss is now imported
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, weight_map):
        ce_pixel_loss = self.ce_loss(logits, targets.long())
        weighted_ce_loss = (ce_pixel_loss * weight_map).mean()
        dice = self.dice_loss(logits, targets)
        return self.alpha * weighted_ce_loss + self.beta * dice

# --- Loss Factory ---

def get_loss(name: str, params: dict, class_weights: np.ndarray, device):
    """
    Loss function factory.
    """
    print(f"   Initializing loss: {name}")

    # Params for losses that need class_weights converted to a list
    loss_params = params.copy()
    if class_weights is not None:
        loss_params['weight'] = class_weights.tolist()

    if name == 'GeminiCombinedLoss':
        return GeminiCombinedLoss(class_weights=class_weights, device=device, **params)
    elif name == 'WeightedCombinedLoss':
        return WeightedCombinedLoss(**loss_params)
    elif name == 'CombinedLoss':
        return CombinedLoss(**loss_params)
    elif name == 'FocalLoss':
        return FocalLoss(**loss_params)
    elif name == 'CBSFNetLoss':
        # Pass device and class_weights directly as they are not in the params dict
        return CBSFNetLoss(class_weights=class_weights, device=device, **params)
    elif name == 'PruningLoss':
        base_loss_name = params.get('base_loss_name')
        base_loss_params = params.get('base_loss_params', {})
        l1_lambda = params.get('l1_lambda', 0.0001)
        
        if not base_loss_name:
            raise ValueError("PruningLoss requires 'base_loss_name' in its parameters.")
            
        # Recursively call get_loss to create the base loss instance
        base_loss = get_loss(base_loss_name, base_loss_params, class_weights, device)
        
        return PruningLoss(base_loss=base_loss, l1_lambda=l1_lambda)
    else:
        raise ValueError(f"Loss function '{name}' not recognized.")