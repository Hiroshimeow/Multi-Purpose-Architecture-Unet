# -*- coding: utf-8 -*-
"""
unet_hsi.py

Script to train a U-Net model for Hyperspectral Semantic Segmentation.
This script focuses on the core training logic and utilizes UnetTool for
logging, checkpointing, and reporting in uniquely timestamped directories.

Usage:
    python unet_hsi.py <base_run_name>

Example:
    python unet_hsi.py unet_run_with_32_filters
"""

# ===== 1. IMPORTS =====
import os
import platform
import sys
from pathlib import Path
import numpy as np
import cv2
import glob
from tqdm import tqdm
from PIL import Image
import h5py
import scipy.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight

# Import our enhanced toolkit
from unet_tool import UnetTool

import warnings
warnings.filterwarnings('ignore')

# ===== 2. CONFIGURATION (Giữ nguyên) =====
def get_config():
    """Returns a dictionary of all hyperparameters and settings."""
    config = {
        'model_architecture': "U-Net",
        'in_channels': 25,
        'num_classes': 5,
        'target_size': (512, 512),
        'batch_size': 16,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'learning_rate': 1e-4,
        'eta_min_cosine': 1e-6,
        'weight_decay': 1e-4,
        'loss_function': 'CombinedLoss',
        'loss_alpha': 0.5,
        'loss_beta': 0.5,
        'data_dir': "Image_dataset",
        'base_output_dir': "training_runs",
    }
    return config

# ===== 3. DATA HANDLING (Giữ nguyên) =====
class HyperspectralDataset(Dataset):
    """Dataset for hyperspectral images, optimized for 25 channels and full-size images."""
    def __init__(self, cube_dir, mask_dir, class_mapping, in_channels=25, target_size=(512, 512), augment=False):
        self.cube_dir = cube_dir
        self.mask_dir = mask_dir
        self.class_mapping = class_mapping
        self.in_channels = in_channels
        self.target_size = target_size
        self.augment = augment
        self.cube_files = sorted(glob.glob(os.path.join(self.cube_dir, "*.mat")))
        if not self.cube_files:
            raise FileNotFoundError(f"No .mat files found in {self.cube_dir}")

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            ToTensorV2(),
        ])
        self.val_transforms = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.cube_files)

    def _load_cube(self, cube_path):
        try:
            mat_data = io.loadmat(cube_path)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not data_keys: raise ValueError(f"No data found in {cube_path}")
            cube = mat_data[data_keys[0]].astype(np.float32)
            if cube.shape[0] == self.in_channels: cube = np.transpose(cube, (1, 2, 0))
        except Exception:
            with h5py.File(cube_path, 'r') as f:
                key = list(f.keys())[0]
                cube = np.array(f[key]).astype(np.float32)
                if cube.shape[0] == self.in_channels: cube = np.transpose(cube, (1, 2, 0))
        return cube

    def _load_mask(self, mask_path):
        return np.array(Image.open(mask_path)).astype(np.uint8)

    def _apply_class_mapping(self, mask):
        mapped_mask = np.zeros_like(mask)
        for original_class, new_class in self.class_mapping.items():
            mapped_mask[mask == original_class] = new_class
        return mapped_mask

    def _normalize_cube(self, cube):
        for i in range(cube.shape[2]):
            channel = cube[:, :, i]
            min_val, max_val = np.min(channel), np.max(channel)
            if max_val > min_val:
                cube[:, :, i] = (channel - min_val) / (max_val - min_val)
        return cube

    def _resize_data(self, cube, mask):
        if cube.shape[:2] != self.target_size:
            cube = cv2.resize(cube, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != self.target_size:
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        return cube, mask

    def __getitem__(self, idx):
        cube_path = self.cube_files[idx]
        base_name = os.path.splitext(os.path.basename(cube_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
        if not os.path.exists(mask_path):
             mask_path = os.path.join(self.mask_dir, f"{base_name.replace('_TC', '')}.png")

        cube = self._load_cube(cube_path)
        mask = self._load_mask(mask_path)
        cube, mask = self._resize_data(cube, mask)
        mask = self._apply_class_mapping(mask)
        cube = self._normalize_cube(cube)

        transforms = self.train_transforms if self.augment else self.val_transforms
        transformed = transforms(image=cube, mask=mask)
        return transformed['image'], transformed['mask'].long()

# ===== 4. MODEL & LOSS FUNCTIONS (Giữ nguyên) =====
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2)
        x = self.up4(x, x1); return self.outc(x)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = DiceLoss()
        self.alpha, self.beta = alpha, beta
    def forward(self, logits, targets):
        return self.alpha * self.ce_loss(logits, targets) + self.beta * self.dice_loss(logits, targets)

# ===== 5. TRAINING & EVALUATION LOOPS (Giữ nguyên) =====
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(masks.cpu().numpy())
    
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_trues = np.concatenate([t.flatten() for t in all_trues])
    
    from sklearn.metrics import accuracy_score, jaccard_score
    accuracy = accuracy_score(flat_trues, flat_preds)
    miou = jaccard_score(flat_trues, flat_preds, average='macro', zero_division=0)
    
    return total_loss / len(loader), accuracy, miou

# ===== 6. MAIN EXECUTION SCRIPT =====
def main():
    """Main function to run the entire training and evaluation pipeline."""
    
    # --- Setup ---
    CONFIG = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(sys.argv) < 2:
        print("Usage: python unet_hsi.py <base_run_name>")
        sys.exit(1)
    base_run_name = sys.argv[1]
    
    class_names = ["Road (tarmac)", "Road marks", "Painted metal", "Sky", "Concrete/stone/brick"]
    
    # Initialize our tool. It will handle creating the unique directory.
    tool = UnetTool(
        base_output_dir=CONFIG['base_output_dir'],
        run_name=base_run_name,
        class_names=class_names
    )
    tool.setup_run(CONFIG)
    
    # --- Data Loading ---
    print("\n1. Preparing Data...")
    data_dir = Path(CONFIG['data_dir'])
    cube_dir, mask_dir = data_dir / "cubes_fl32", data_dir / "labels"
    class_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 0: 0, 3: 2, 7: 2, 9: 2, 10: 4}
    
    full_dataset = HyperspectralDataset(str(cube_dir), str(mask_dir), class_mapping, 
                                        in_channels=CONFIG['in_channels'], target_size=CONFIG['target_size'])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_dataset.dataset.augment = True
    val_dataset.dataset.augment = False
    
    num_workers = min(os.cpu_count(), 8) if platform.system() != 'Windows' else 0
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"✓ Data ready: {len(train_dataset)} train, {len(val_dataset)} validation samples.")

    # --- Model & Training Components Setup ---
    print("\n2. Setting up Model and Training Components...")
    model = UNet(in_channels=CONFIG['in_channels'], num_classes=CONFIG['num_classes']).to(device)
    
    all_masks = np.concatenate([mask.numpy().flatten() for _, mask in tqdm(train_dataset, desc="Sampling for weights")])
    class_weights = torch.FloatTensor(compute_class_weight('balanced', classes=np.arange(CONFIG['num_classes']), y=all_masks)).to(device)
    
    criterion = CombinedLoss(weight=class_weights, alpha=CONFIG['loss_alpha'], beta=CONFIG['loss_beta'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=CONFIG['eta_min_cosine'])
    scaler = GradScaler()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized: {CONFIG['model_architecture']} with {num_params / 1e6:.2f}M parameters.")
    print(f"✓ Loss: {CONFIG['loss_function']}, Optimizer: {CONFIG['optimizer']}, Scheduler: {CONFIG['scheduler']}")

    # --- Training Loop ---
    print("\n3. Starting Training...")
    best_miou = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        metrics = {
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_acc': val_acc, 'val_miou': val_miou,
            'lr': optimizer.param_groups[0]['lr']
        }
        tool.log_epoch(metrics)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val mIoU: {val_miou:.4f}")
        
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou
            patience_counter = 0
            print(f"✨ New best mIoU! {val_miou:.4f}. Saving model...")
        else:
            patience_counter += 1
        
        tool.save_checkpoint(model, optimizer, epoch + 1, is_best)
            
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement.")
            break
            
    print(f"\n🏁 Training Finished! Best Validation mIoU: {best_miou:.4f}")

    # --- Final Evaluation & Analysis ---
    print("\n4. Final Evaluation and Analysis...")
    print("   Loading best model for detailed report...")
    model = tool.load_model_for_inference(model, device)
    
    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Final Evaluation"):
            outputs = model(images.to(device))
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_trues.append(masks.numpy())
            
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_trues = np.concatenate([t.flatten() for t in all_trues])
    
    tool.generate_final_report(best_miou, flat_trues, flat_preds)
    
    print(f"\n✅ Analysis complete. All results saved to '{tool.output_dir}' directory.")

if __name__ == '__main__':
    main()