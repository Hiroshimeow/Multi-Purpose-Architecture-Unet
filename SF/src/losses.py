# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Loss Components ---

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

# --- Combined and Special Loss Functions ---

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5, beta=0.5, **kwargs):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight) if weight is not None else None)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, model_output, targets):
        # Handle dictionary input for compatibility
        if isinstance(model_output, dict):
            logits = model_output['segmentation']
        else:
            logits = model_output
            
        ce = self.ce_loss(logits, targets.long())
        dice = self.dice_loss(logits, targets)
        return self.alpha * ce + self.beta * dice

class PruningLoss(nn.Module):
    """
    A wrapper loss for channel pruning experiments.
    Combines a base segmentation loss with an L1 penalty on channel weights.
    """
    def __init__(self, base_loss_name, base_loss_params, l1_lambda, **kwargs):
        super().__init__()
        self.l1_lambda = l1_lambda
        # Create the base loss function, passing through kwargs like class_weights
        self.base_loss = get_loss(name=base_loss_name, params=base_loss_params, **kwargs)

    def forward(self, model_output, targets):
        # The model is expected to return a dictionary
        if not isinstance(model_output, dict) or 'segmentation' not in model_output or 'channel_weights' not in model_output:
            raise TypeError(
                "PruningLoss requires the model to return a dictionary "
                "containing 'segmentation' and 'channel_weights'."
            )

        segmentation_logits = model_output['segmentation']
        channel_weights = model_output['channel_weights']

        # Calculate the primary segmentation loss
        # The base loss now also needs to handle the dict, so we pass only the logits
        segmentation_loss = self.base_loss(segmentation_logits, targets)

        # Calculate L1 penalty on the absolute channel weights to encourage sparsity
        l1_penalty = torch.norm(channel_weights, p=1)

        # Return the combined loss
        return segmentation_loss + self.l1_lambda * l1_penalty

# --- Loss Factory ---
def get_loss(name: str, params: dict, class_weights: np.ndarray = None, device: torch.device = None):
    """
    Factory function to create a loss instance.
    """
    print(f"   Initializing loss: {name}")
    
    kwargs = {}
    if class_weights is not None:
        kwargs['weight'] = class_weights
    if device is not None:
        kwargs['device'] = device

    if name == 'CombinedLoss':
        return CombinedLoss(**params, **kwargs)
    elif name == 'FocalLoss':
        return FocalLoss(**params, **kwargs)
    elif name == 'PruningLoss':
        # For PruningLoss, params dict has a special structure.
        # We pass the kwargs (weights, device) down to the base loss constructor.
        return PruningLoss(**params, **kwargs)
    else:
        raise ValueError(f"Loss function '{name}' not recognized.")
