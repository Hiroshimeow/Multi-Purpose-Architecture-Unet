# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# --- Các thành phần Loss cơ bản ---

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

# --- Các hàm Loss kết hợp ---

class WeightedCombinedLoss(nn.Module):
    """
    [GIỮ NGUYÊN] Hàm loss gốc, tập trung vào ranh giới.
    Sử dụng CrossEntropy có trọng số pixel (pixel-wise weight map).
    """
    def __init__(self, weight=None, alpha=1.0, beta=0.5):
        super(WeightedCombinedLoss, self).__init__()
        # CRITICAL: reduction='none' để tính loss cho từng pixel
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weight) if weight is not None else None,
            reduction='none'
        )
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, weight_map):
        # --- Weighted Cross Entropy ---
        ce_pixel_loss = self.ce_loss(logits, targets.long())
        # Nhân loss của từng pixel với trọng số tương ứng
        weighted_ce_loss = (ce_pixel_loss * weight_map).mean()

        # --- Dice Loss (không cần trọng số pixel) ---
        dice = self.dice_loss(logits, targets)
        
        return self.alpha * weighted_ce_loss + self.beta * dice
    
class CombinedLoss(nn.Module):
    """
    [MỚI] Hàm loss đơn giản hơn, không sử dụng weight map.
    Chỉ kết hợp CrossEntropy (có trọng số lớp) và Dice Loss.
    Đây là phiên bản ổn định để dùng với chiến lược cắt patch.
    """
    def __init__(self, weight=None, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight) if weight is not None else None)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        # Hàm forward này không nhận weight_map
        ce = self.ce_loss(logits, targets.long())
        dice = self.dice_loss(logits, targets)
        return self.alpha * ce + self.beta * dice

# --- [MỚI] Loss Factory ---
def get_loss(name: str, params: dict, class_weights: np.ndarray = None):
    """
    Hàm factory để tạo instance của một hàm loss dựa trên tên và tham số.
    """
    print(f"   Initializing loss: {name}")
    
    # Truyền trọng số lớp vào params nếu có
    if class_weights is not None:
        params['weight'] = class_weights.tolist()

    if name == 'WeightedCombinedLoss':
        return WeightedCombinedLoss(**params)
    elif name == 'CombinedLoss':
        return CombinedLoss(**params)
    else:
        raise ValueError(f"Loss function '{name}' not recognized.")