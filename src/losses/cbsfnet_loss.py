'''
Multi-task loss function for the Contextual Boundary-Aware Spectral Fusion Network (CB-SFNet).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel

from .base_losses import CombinedLoss

class CBSFNetLoss(nn.Module):
    def __init__(self, class_weights, alpha=1.0, beta=0.5, gamma=0.1, delta=0.1, device='cuda'):
        """
        Initializes the multi-task loss for CB-SFNet.

        Args:
            class_weights (np.ndarray): Weights for each class for the segmentation loss.
            alpha (float): Weight for the main segmentation loss (L_seg).
            beta (float): Weight for the boundary prediction loss (L_boundary).
            gamma (float): Weight for the contrastive loss (L_contrast).
            delta (float): Weight for the physics-informed consistency loss (L_physics).
            device (str): The device to run calculations on.
        """
        super(CBSFNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.device = device

        # 1. Segmentation Loss (L_seg)
        self.segmentation_loss = CombinedLoss(weight=class_weights, alpha=0.5, beta=0.5) # Internal alpha/beta for Dice/CE

        # 2. Boundary Loss (L_boundary)
        self.boundary_loss = nn.BCEWithLogitsLoss()

    def get_ground_truth_boundary(self, mask):
        """
        Generates a boundary mask from a segmentation mask on the fly.
        Args:
            mask (torch.Tensor): Ground truth segmentation mask, shape (B, H, W).
        Returns:
            torch.Tensor: Boundary mask, shape (B, 1, H, W).
        """
        # Convert mask to one-hot, then take the max along the class dimension to get a binary map for any class vs. background
        # This is a simplification. A better approach might be to find edges between any two different classes.
        # For now, we find edges of all segmented regions.
        binary_mask = (mask > 0).float().unsqueeze(1) # Shape: (B, 1, H, W)
        # Use kornia's sobel filter to find edges
        boundary = sobel(binary_mask)
        boundary = (boundary > 0.1).float() # Binarize the result
        return boundary

    def contrastive_loss(self, features, mask):
        """
        Simplified implementation of Cross-Scale Contextual Contrast Learning (CSCCL).
        Pulls features from the same class together and pushes features from different classes apart.
        """
        # This is a highly simplified version. A real implementation would be much more complex.
        # For now, we return a zero loss to avoid introducing instability during initial testing.
        return torch.tensor(0.0, device=self.device)

    def physics_consistency_loss(self, seg_logits, mask):
        """
        Simplified implementation of Physics-Informed Spectral Consistency Module (PISCM).
        This encourages smoothness of predictions within an object.
        """
        # Get segmentation probabilities
        seg_probs = torch.softmax(seg_logits, dim=1)
        
        # Calculate image gradients (smoothness penalty)
        grad_y = torch.abs(seg_probs[:, :, 1:, :] - seg_probs[:, :, :-1, :])
        grad_x = torch.abs(seg_probs[:, :, :, 1:] - seg_probs[:, :, :, :-1])
        
        # Create masks for where the gradients should be low (i.e., inside objects)
        # We can use the ground truth mask for this.
        mask_y = mask[:, 1:, :].unsqueeze(1)
        mask_x = mask[:, :, 1:].unsqueeze(1)
        
        # Penalize gradients that are not on a boundary
        # A simple approximation: penalize all gradients, but more strongly where there is no GT boundary
        gt_boundary = self.get_ground_truth_boundary(mask)
        gt_boundary_y = gt_boundary[:, :, 1:, :]
        gt_boundary_x = gt_boundary[:, :, :, 1:]

        consistency_loss = (grad_y * (1 - gt_boundary_y)).mean() + (grad_x * (1 - gt_boundary_x)).mean()
        return consistency_loss

    def forward(self, outputs, targets):
        """
        Calculates the total loss from the model's dictionary of outputs.
        Args:
            outputs (dict): A dictionary from the model, containing 'segmentation', 'boundary', etc.
            targets (torch.Tensor): The ground truth segmentation mask.
        """
        # Move targets to the correct device
        targets = targets.to(self.device)

        # 1. Segmentation Loss
        l_seg = self.segmentation_loss(outputs['segmentation'], targets)

        # 2. Boundary Loss
        gt_boundary = self.get_ground_truth_boundary(targets)
        l_boundary = self.boundary_loss(outputs['boundary'], gt_boundary)

        # 3. Contrastive Loss (Simplified)
        l_contrast = self.contrastive_loss(outputs['contrast_features'], targets)

        # 4. Physics-Informed Consistency Loss (Simplified)
        l_physics = self.physics_consistency_loss(outputs['segmentation'], targets)

        # Combine losses
        total_loss = (self.alpha * l_seg) + (self.beta * l_boundary) + (self.gamma * l_contrast) + (self.delta * l_physics)

        # TODO: Log individual loss components for better monitoring
        # For now, we just return the total loss.
        return total_loss
