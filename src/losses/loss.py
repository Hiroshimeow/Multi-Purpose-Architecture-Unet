
# src/losses/gemini_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=1)
        num_classes = probas.shape[1]
        dice_total = 0
        
        for i in range(num_classes):
            probas_i = probas[:, i, :, :]
            targets_i = (targets == i).float()
            intersection = (probas_i * targets_i).sum()
            union = probas_i.sum() + targets_i.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_total += (1 - dice)
            
        return dice_total / num_classes

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss

        if self.alpha is not None:
            # Move alpha to the same device as targets
            alpha_t = self.alpha[targets].to(targets.device)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PruningLoss(nn.Module):
    def __init__(self, base_loss, l1_lambda=0.0001):
        super(PruningLoss, self).__init__()
        self.base_loss = base_loss
        self.l1_lambda = l1_lambda

    def forward(self, outputs, targets):
        # The model returns a dictionary, so we access the keys directly
        seg_outputs = outputs['segmentation']
        channel_weights = outputs['channel_weights']
        
        # Calculate the base segmentation loss
        segmentation_loss = self.base_loss(seg_outputs, targets)
        
        # Calculate L1 penalty on channel weights
        l1_penalty = torch.norm(channel_weights, p=1)
        
        # Combine the losses
        total_loss = segmentation_loss + self.l1_lambda * l1_penalty
        
        return total_loss

class GeminiCombinedLoss(nn.Module):
    def __init__(self, class_weights, device, focal_w=0.4, dice_w=0.6, ds_weights=None, **kwargs):
        super(GeminiCombinedLoss, self).__init__()
        if ds_weights is None:
            ds_weights = [0.4, 0.2] # Default values
            
        self.focal_loss = FocalLoss(alpha=torch.tensor(class_weights, dtype=torch.float32).to(device), gamma=2)
        self.dice_loss = DiceLoss()
        self.w = [focal_w, dice_w]
        self.ds_weights = ds_weights

    def _calculate_loss(self, logits, targets):
        # Ensure logits are float for loss calculation
        logits = logits.float()
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.w[0] * focal + self.w[1] * dice

    def forward(self, outputs, targets):
        # The model returns a tuple: (list_of_outputs, band_weights)
        # We only need the list_of_outputs for the loss
        if isinstance(outputs, tuple):
            list_of_outputs = outputs[0]
        else: # Fallback if the model just returns the list
            list_of_outputs = outputs

        main_output = list_of_outputs[0]
        ds_outputs = list_of_outputs[1:]

        main_loss = self._calculate_loss(main_output, targets)
        total_loss = main_loss

        for i, ds_out in enumerate(ds_outputs):
            # Resize target mask to match the size of the deep supervision output
            ds_target = F.interpolate(targets.unsqueeze(1).float(), size=ds_out.shape[2:], mode='nearest').squeeze(1).long()
            ds_loss = self._calculate_loss(ds_out, ds_target)
            
            # Make sure we don't go out of bounds for weights
            if i < len(self.ds_weights):
                total_loss += self.ds_weights[i] * ds_loss

        return total_loss
