"""
=================================================================================
*** GEMINI-ENHANCED UNET FOR HYPERSPECTRAL IMAGE SEGMENTATION (geminii_learnable_unet.py) ***
=================================================================================
Mô tả:
Script huấn luyện được nâng cấp bởi Gemini, tập trung vào việc tối ưu hóa và
cải thiện độ chính xác (mIoU) cho mô hình UNet-CBAM với cơ chế "Learnable Selector".

*** CÁC CẢI TIẾN CHÍNH (do Gemini thực hiện) ***
1.  **[KIẾN TRÚC] Deep Supervision:** Thêm các đầu ra phụ ở các tầng giải mã để
    chống vanishing gradient và cải thiện khả năng học sâu.
2.  **[HÀM LOSS] Focal Loss + Dice Loss:** Thay thế Cross-Entropy bằng Focal Loss
    để tập trung vào các mẫu khó, giải quyết tốt hơn vấn đề mất cân bằng lớp.
3.  **[DỮ LIỆU] Dataset-level Normalization:** Chuẩn hóa dữ liệu dựa trên mean/std
    của toàn bộ tập huấn luyện, giúp model ổn định và hội tụ tốt hơn.
4.  **[CODE] Refactor Kiến Trúc UNet:** Tái cấu trúc lại các module của UNet
    (encoder, decoder) để code trở nên rõ ràng, dễ đọc và dễ bảo trì hơn.
5.  **[HUẤN LUYỆN] Tích hợp AMP (Automatic Mixed Precision):** Sử dụng AMP để
    tăng tốc độ huấn luyện và giảm bộ nhớ GPU mà không ảnh hưởng nhiều đến độ chính xác.

Cách sử dụng:
1.  Cài đặt thư viện:
    pip install torch torchvision scikit-learn scipy tqdm pandas matplotlib albumentations thop
2.  Đảm bảo cấu trúc thư mục:
    /your_project_folder/
    |-- geminii_learnable_unet.py (file này)
    |-- Image_dataset/
        |-- cubes_fl32/
        |-- labels/
3.  Chạy script:
    source /data1/.venv/bin/activate
    python geminii_learnable_unet.py
"""

# =================================================================================
# Block 1: Imports & Setup
# =================================================================================
import os
import glob
import argparse
import logging
from datetime import datetime
import numpy as np
import scipy.io as io
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from thop import profile
import warnings

warnings.filterwarnings('ignore')

# =================================================================================
# Block 2: Configuration
# =================================================================================
CONFIG = {
    # --- Path Configuration ---
    'data_path': './Image_dataset',
    'output_dir': './results_gemini', # Thư mục output mới

    # --- Data & Class Configuration ---
    'num_classes': 5,
    'class_names': ["Road", "RoadLine", "Vegetation", "Sky", "NoDrivable"],
    'class_mapping': {1: 0, 2: 1, 3: 2, 5: 3, 0: 4, 4: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4},

    # --- Training Hyperparameters ---
    'target_size': (224, 416),
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'batch_size': 8,
    'epochs': 150,
    'early_stopping_patience': 20,

    # --- [CẢI TIẾN] Loss Configuration ---
    'loss_dice_weight': 0.6,
    'loss_focal_weight': 0.4, # Thay CE bằng Focal
    'deep_supervision_weights': [0.4, 0.2], # Trọng số cho các đầu ra phụ

    # --- Model Architecture ---
    'model_initial_filters': 64,
    'model_encoder_depth': 5,
    'model_dropout_rate': 0.3,
}

# =================================================================================
# Block 3: Data Augmentation
# =================================================================================
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.3),
            A.GridDistortion(p=0.2),
            A.ElasticTransform(p=0.2), # [CẢI TIẾN] Thêm ElasticTransform
            ToTensorV2(),
        ])
    else:
        return A.Compose([ToTensorV2()])

# =================================================================================
# Block 4: Hyperspectral Dataset Class
# =================================================================================
class HyperspectralDataset(Dataset):
    # [CẢI TIẾN] Thêm mean và std của dataset
    def __init__(self, cube_files, mask_dir, class_mapping, target_size, transform=None, dataset_mean=None, dataset_std=None):
        self.cube_files = cube_files
        self.mask_dir = mask_dir
        self.class_mapping = class_mapping
        self.target_size = target_size
        self.transform = transform
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def __len__(self):
        return len(self.cube_files)

    def _load_cube(self, cube_path):
        mat_data = io.loadmat(cube_path)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        cube = mat_data[data_keys[0]].astype(np.float32)
        return np.transpose(cube, (1, 2, 0)) if cube.shape[0] == 25 else cube

    def _apply_class_mapping(self, mask):
        mapped_mask = np.zeros_like(mask, dtype=np.uint8)
        for original_class, new_class in self.class_mapping.items():
            mapped_mask[mask == original_class] = new_class
        return mapped_mask

    def __getitem__(self, idx):
        cube_path = self.cube_files[idx]
        try:
            base_name = os.path.splitext(os.path.basename(cube_path))[0]
            mask_path = os.path.join(self.mask_dir, f"{base_name.replace('_TC', '')}.png")

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found for cube: {cube_path} -> Expected at: {mask_path}")

            cube = self._load_cube(cube_path)
            mask = self._apply_class_mapping(np.array(Image.open(mask_path)))

            cube = cv2.resize(cube, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

            if self.transform:
                augmented = self.transform(image=cube, mask=mask)
                cube_tensor = augmented['image']
                mask_tensor = augmented['mask'].long()
            else:
                cube_tensor = torch.from_numpy(cube.transpose(2, 0, 1))
                mask_tensor = torch.from_numpy(mask).long()

            # [CẢI TIẾN] Sử dụng mean/std của toàn dataset nếu có
            if self.dataset_mean is not None and self.dataset_std is not None:
                cube_tensor = (cube_tensor - self.dataset_mean) / self.dataset_std
            else: # Fallback về chuẩn hóa trên từng ảnh
                mean = torch.mean(cube_tensor, dim=(1, 2), keepdim=True)
                std = torch.std(cube_tensor, dim=(1, 2), keepdim=True) + 1e-8
                cube_tensor = (cube_tensor - mean) / std

            return cube_tensor, mask_tensor

        except Exception as e:
            print(f"ERROR loading sample {cube_path}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

# [CẢI TIẾN] Hàm tính mean/std cho toàn bộ dataset
def calculate_dataset_stats(dataset_loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    pbar = tqdm(dataset_loader, desc='Calculating Dataset Stats')
    for data, _ in pbar:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    return mean.view(25, 1, 1), std.view(25, 1, 1)

# =================================================================================
# Block 5: Learnable UNet-CBAM Model Architecture (Refactored & with Deep Supervision)
# =================================================================================
class BandSelector(nn.Module):
    def __init__(self, in_channels):
        super(BandSelector, self).__init__()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels, 1, bias=False), nn.Sigmoid())
    def forward(self, x):
        weights = self.attention(x); return x * weights, weights.squeeze()

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sa = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.ca(x); max_out = self.ca(x); ca_out = self.sigmoid(avg_out + max_out)
        x = x * ca_out
        avg_out = torch.mean(x, dim=1, keepdim=True); max_out, _ = torch.max(x, dim=1, keepdim=True); sa_in = torch.cat([avg_out, max_out], dim=1)
        sa_out = self.sigmoid(self.sa(sa_in))
        x = x * sa_out
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(); self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.conv(x)

# [CẢI TIẾN] Kiến trúc UNet được refactor và thêm Deep Supervision
class LearnableUNetCBAM(nn.Module):
    def __init__(self, n_channels, n_classes, initial_filters=64, depth=5, dropout_rate=0.3):
        super(LearnableUNetCBAM, self).__init__()
        self.depth = depth
        self.band_selector = BandSelector(n_channels)

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        in_ch = n_channels
        for i in range(depth):
            out_ch = initial_filters * (2**i)
            self.encoders.append(
                nn.Sequential(
                    DoubleConv(in_ch, out_ch),
                    CBAM(out_ch),
                    nn.MaxPool2d(2),
                    nn.Dropout(dropout_rate)
                )
            )
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            DoubleConv(in_ch, in_ch * 2),
            CBAM(in_ch * 2)
        )
        in_ch *= 2

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        self.deep_supervision_outputs = nn.ModuleList() # For deep supervision
        for i in range(depth - 1, -1, -1):
            up_ch = initial_filters * (2**i)
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2),
                    DoubleConv(up_ch * 2, up_ch), # Skip connection doubles channels
                    CBAM(up_ch)
                )
            )
            in_ch = up_ch
            # [CẢI TIẾN] Add deep supervision outputs for higher-resolution decoder stages
            if i > depth - 3: # Add for last 2 decoder stages
                 self.deep_supervision_outputs.append(OutConv(up_ch, n_classes))

        self.outc = OutConv(initial_filters, n_classes)

    def forward(self, x):
        x, band_weights = self.band_selector(x)
        skip_connections = []
        ds_outputs = []

        # Encoder path
        for i in range(self.depth):
            x = self.encoders[i][0](x) # DoubleConv
            x = self.encoders[i][1](x) # CBAM
            skip_connections.append(x)
            x = self.encoders[i][2](x) # MaxPool
            x = self.encoders[i][3](x) # Dropout

        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1] # Reverse for decoder
        for i in range(self.depth):
            x = self.decoders[i][0](x) # ConvTranspose
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[i][1](x) # DoubleConv
            x = self.decoders[i][2](x) # CBAM

            # [CẢI TIẾN] Get deep supervision outputs
            if i >= self.depth - len(self.deep_supervision_outputs):
                ds_idx = i - (self.depth - len(self.deep_supervision_outputs))
                ds_out = self.deep_supervision_outputs[ds_idx](x)
                ds_outputs.append(ds_out)

        main_output = self.outc(x)
        ds_outputs.reverse() # Align with weights (high-res first)

        # Return main output, deep supervision outputs, and band weights
        return [main_output] + ds_outputs, band_weights

# =================================================================================
# Block 6: Advanced Loss Function
# =================================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=1); num_classes = probas.shape[1]; dice_total = 0
        for i in range(num_classes):
            probas_i = probas[:, i, :, :]; targets_i = (targets == i).float()
            intersection = (probas_i * targets_i).sum()
            union = probas_i.sum() + targets_i.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_total += (1 - dice)
        return dice_total / num_classes

# [CẢI TIẾN] Focal Loss
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
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# [CẢI TIẾN] Combined Loss with Deep Supervision support
class CombinedLoss(nn.Module):
    def __init__(self, class_weights, focal_w, dice_w, ds_weights, device):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=torch.tensor(class_weights, dtype=torch.float32).to(device), gamma=2)
        self.dice_loss = DiceLoss()
        self.w = [focal_w, dice_w]
        self.ds_weights = ds_weights

    def _calculate_loss(self, logits, targets):
        focal = self.focal_loss(logits.float(), targets)
        dice = self.dice_loss(logits.float(), targets)
        return self.w[0] * focal + self.w[1] * dice

    def forward(self, outputs, targets):
        main_output = outputs[0]
        ds_outputs = outputs[1:]

        # Resize target for deep supervision outputs
        main_loss = self._calculate_loss(main_output, targets)
        total_loss = main_loss

        for i, ds_out in enumerate(ds_outputs):
            ds_target = F.interpolate(targets.unsqueeze(1).float(), size=ds_out.shape[2:], mode='nearest').squeeze(1).long()
            ds_loss = self._calculate_loss(ds_out, ds_target)
            total_loss += self.ds_weights[i] * ds_loss

        return total_loss

# =================================================================================
# Block 7: Training & Evaluation Utilities
# =================================================================================
def calculate_metrics(preds, labels, num_classes):
    iou_scores = []
    preds = torch.argmax(preds, dim=1).view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        iou_scores.append(intersection / union if union > 0 else float('nan'))
    return iou_scores

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        if images is None or masks is None: continue
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs, _ = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_ious = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                outputs, _ = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()
            # Use only the main output for metric calculation
            iou = calculate_metrics(outputs[0].cpu(), masks.cpu(), num_classes)
            all_ious.append(iou)

    avg_loss = total_loss / len(loader)
    mean_ious = np.nanmean(np.array(all_ious), axis=0)
    miou = np.nanmean(mean_ious)
    return avg_loss, miou, mean_ious

# =================================================================================
# Block 8: Main Execution Logic
# =================================================================================
def main(args):
    # --- 1. Setup Logging and Directories ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Gemini_UNET_{CONFIG['num_classes']}cls_{timestamp}"
    output_dir = os.path.join(CONFIG['output_dir'], experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join(output_dir, 'training.log')), logging.StreamHandler()])
    logging.info(f"Experiment started: {experiment_name}")
    logging.info(f"Results will be saved to: {output_dir}")
    logging.info(f"Using configuration: {CONFIG}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 2. Load Data and Create Dataloaders ---
    logging.info("Loading and splitting data...")
    cube_dir = os.path.join(CONFIG['data_path'], "cubes_fl32")
    mask_dir = os.path.join(CONFIG['data_path'], "labels")
    all_files = sorted(glob.glob(os.path.join(cube_dir, "*.mat")))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # [CẢI TIẾN] Calculate dataset stats
    logging.info("Calculating dataset statistics for normalization...")
    stats_dataset = HyperspectralDataset(train_files, mask_dir, CONFIG['class_mapping'], CONFIG['target_size'], transform=get_transforms(is_train=False))
    stats_loader = DataLoader(stats_dataset, batch_size=CONFIG['batch_size'], num_workers=4, collate_fn=collate_fn)
    dataset_mean, dataset_std = calculate_dataset_stats(stats_loader)
    logging.info(f"Dataset Mean and Std calculated.")

    train_dataset = HyperspectralDataset(train_files, mask_dir, CONFIG['class_mapping'], CONFIG['target_size'], get_transforms(is_train=True), dataset_mean, dataset_std)
    val_dataset = HyperspectralDataset(val_files, mask_dir, CONFIG['class_mapping'], CONFIG['target_size'], get_transforms(is_train=False), dataset_mean, dataset_std)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    logging.info(f"Data loaded: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

    # --- 3. Model, Loss, Optimizer Setup ---
    logging.info("Setting up model, loss, and optimizer...")
    model = LearnableUNetCBAM(n_channels=25, n_classes=CONFIG['num_classes'], initial_filters=CONFIG['model_initial_filters'], depth=CONFIG['model_encoder_depth'], dropout_rate=CONFIG['model_dropout_rate']).to(device)

    all_masks = [train_dataset[i][1].numpy().flatten() for i in range(len(train_dataset)) if train_dataset[i] is not None]
    class_weights = compute_class_weight('balanced', classes=np.arange(CONFIG['num_classes']), y=np.concatenate(all_masks))
    logging.info(f"Class weights for Focal loss: {class_weights}")

    criterion = CombinedLoss(class_weights, CONFIG['loss_focal_weight'], CONFIG['loss_dice_weight'], CONFIG['deep_supervision_weights'], device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6) # [CẢI TIẾN] Scheduler
    scaler = torch.cuda.amp.GradScaler()

    # --- 4. Pre-computation for Benchmarking ---
    logging.info("Calculating model complexity (GFLOPS, Parameters)...")
    dummy_input = torch.randn(1, 25, *CONFIG['target_size']).to(device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    gflops = (macs * 2) / 1e9; params_m = params / 1e6
    logging.info(f"Model Parameters: {params_m:.2f}M"); logging.info(f"GFLOPS: {gflops:.2f}")

    # --- 5. Training Loop ---
    logging.info("Starting training loop...")
    best_miou = 0.0; patience_counter = 0; history = []
    for epoch in range(1, CONFIG['epochs'] + 1):
        logging.info(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_miou, val_ious_per_class = evaluate(model, val_loader, criterion, device, CONFIG['num_classes'])
        scheduler.step()

        log_entry = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_miou': val_miou}
        logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
        for i, iou in enumerate(val_ious_per_class):
            class_name = CONFIG['class_names'][i]
            logging.info(f"  - {class_name} IoU: {iou:.4f}"); log_entry[f'iou_{class_name}'] = iou
        history.append(log_entry)
        pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_log.csv'), index=False)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            logging.info(f"✨ New best model saved with mIoU: {best_miou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= CONFIG['early_stopping_patience']:
            logging.info("Early stopping triggered."); break

    # --- 6. Final Analysis and Reporting ---
    logging.info("Training finished. Performing final analysis...")
    # Load best model for final analysis
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    model.eval()

    # Get final band weights
    with torch.no_grad():
        _, final_band_weights = model(dummy_input)
    final_band_weights = final_band_weights.cpu().numpy().flatten()

    # Plot and save band weights
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, 26), final_band_weights)
    plt.title('Learned Band Importance (from BandSelector)')
    plt.xlabel('Band Number')
    plt.ylabel('Attention Weight')
    plt.xticks(range(1, 26))
    plt.grid(axis='y', linestyle='--')
    band_weights_path = os.path.join(output_dir, 'band_weights.png')
    plt.savefig(band_weights_path)
    plt.close()
    logging.info(f"Band weights visualization saved to {band_weights_path}")

    # --- Create Final Summary Report ---
    final_metrics = {
        'Method': 'Gemini-UNet',
        'mIoU': best_miou,
        'Parameters (M)': params_m,
        'GFLOPS': gflops,
        'mIoU / GFLOPS': best_miou / gflops if gflops > 0 else 0,
        'mIoU / Params (M)': best_miou / params_m if params_m > 0 else 0,
    }
    summary_df = pd.DataFrame([final_metrics])
    summary_path = os.path.join(output_dir, 'summary_report.csv')
    summary_df.to_csv(summary_path, index=False)

    logging.info("\n--- FINAL SUMMARY REPORT ---")
    logging.info(f"\n{summary_df.to_string()}")
    logging.info(f"Summary report saved to {summary_path}")

    # Plot mIoU vs GFLOPS
    plt.figure(figsize=(8, 6))
    plt.scatter(summary_df['GFLOPS'], summary_df['mIoU'], s=100, label=summary_df['Method'][0])
    plt.title('Performance vs. Computational Cost')
    plt.xlabel('GFLOPS (Lower is Better)')
    plt.ylabel('mIoU (Higher is Better)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'performance_plot.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Performance plot saved to {plot_path}")
    logging.info("--- Experiment Finished Successfully ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Gemini-enhanced Learnable UNet-CBAM model.")
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'], help='Number of training epochs')
    args = parser.parse_args([])
    CONFIG['epochs'] = args.epochs
    main(args)
