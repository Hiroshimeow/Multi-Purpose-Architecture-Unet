# src/losses/base_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5, beta=0.5, ds_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight) if weight is not None else None)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta
        self.ds_weights = ds_weights if ds_weights else [0.4, 0.2] # Default weights for DS outputs

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            # Handle dictionary output from models like ViT_UNet
            outputs = [outputs['segmentation']]
        elif not isinstance(outputs, list):
            outputs = [outputs]

        total_loss = 0
        main_loss = self.alpha * self.ce_loss(outputs[0], targets.long()) + self.beta * self.dice_loss(outputs[0], targets)
        total_loss += main_loss

        if len(outputs) > 1:
            for i, (output, weight) in enumerate(zip(outputs[1:], self.ds_weights)):
                target_downsampled = F.interpolate(targets.unsqueeze(1).float(), size=output.shape[2:], mode='nearest').squeeze(1)
                ds_loss = self.alpha * self.ce_loss(output, target_downsampled.long()) + self.beta * self.dice_loss(output, target_downsampled)
                total_loss += weight * ds_loss
        
        return total_loss
