# run.py
# Hợp nhất từ các file run*.py để tạo ra một script huấn luyện duy nhất, linh hoạt.
import argparse
import yaml
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import platform
import glob
import random
import time
import h5py
import scipy.io as io

from src.models import get_model
from src.datasets import TiledHyperspectralDataset
from src.losses import get_loss
from src.experiment_manager import ExperimentManager
from src.trainer import Trainer

# Helper to load band rankings
def load_band_rankings(file_path="band_rankings.txt"):
    rankings = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                method, bands_str = line.strip().split(":")
                rankings[method.strip()] = [int(b) for b in bands_str.split(",")]
    return rankings

def calculate_zscore_stats(file_paths: list, in_channels: int, selected_bands: list = None):
    """
    Tính toán giá trị trung bình và độ lệch chuẩn trên tập huấn luyện để chuẩn hóa Z-score.
    Bổ sung tham số selected_bands để chỉ tính toán trên các kênh được chọn.
    """
    from torch.utils.data import Dataset
    import torch

    print("Calculating Z-score statistics on the training set...")

    class StatsDataset(Dataset):
        def __init__(self, files, channels, selected_bands_indices=None):
            self.files = files
            self.channels = channels
            self.selected_bands_indices = selected_bands_indices
        def __len__(self):
            return len(self.files)
        def __getitem__(self, idx):
            try:
                # Thử đọc bằng h5py trước
                with h5py.File(self.files[idx], 'r') as f:
                    data_keys = list(f.keys())
                    if not data_keys:
                        return None
                    cube = np.array(f[data_keys[0]]).astype(np.float32)
            except (OSError, IOError):
                # Nếu thất bại, thử đọc bằng scipy.io
                try:
                    mat_data = io.loadmat(self.files[idx])
                    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    if not data_keys:
                        return None
                    cube = mat_data[data_keys[0]].astype(np.float32)
                except Exception as e:
                    print(f"Warning: Could not load {self.files[idx]} with scipy. Error: {e}")
                    return None
            except Exception as e:
                print(f"Warning: An unexpected error occurred with {self.files[idx]}. Error: {e}")
                return None

            # Chuẩn hóa chiều của cube về (C, H, W) nếu cần
            if cube.ndim == 3 and cube.shape[2] == 25 and cube.shape[0] != 25: # Assuming 25 original channels
                cube = np.transpose(cube, (2, 0, 1))
            
            # Thực hiện chọn băng tần nếu có
            if self.selected_bands_indices is not None:
                cube = cube[self.selected_bands_indices, :, :]

            # Kiểm tra lại lần cuối
            if cube.shape[0] != self.channels:
                print(f"Warning: Cube channel mismatch after selection: Expected {self.channels}, got {cube.shape[0]}")
                return None

            return torch.from_numpy(cube)

    dataset = StatsDataset(file_paths, in_channels, selected_bands)

    def collate_fn_skip_none(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        return torch.utils.data.default_collate(batch)

    # Tăng batch_size và num_workers để tính toán nhanh hơn
    loader = DataLoader(dataset, batch_size=16, num_workers=min(os.cpu_count(), 8),
                        shuffle=False, collate_fn=collate_fn_skip_none)

    count = 0
    mean = torch.zeros(in_channels)
    m2 = torch.zeros(in_channels)

    for cube_batch in tqdm(loader, desc="Calculating Stats"):
        if cube_batch is None:
            continue

        # cube_batch: (B, C, H, W)
        pixels = cube_batch.permute(0, 2, 3, 1).reshape(-1, in_channels)
        n = pixels.shape[0]
        if n == 0:
            continue

        batch_mean = pixels.mean(dim=0)
        batch_var = pixels.var(dim=0, unbiased=False)

        delta = batch_mean - mean
        mean += delta * n / (count + n)
        m2 += batch_var * n + delta**2 * count * n / (count + n)
        count += n

    if count < 2:
        print("Warning: Not enough data to calculate standard deviation. Returning std=1.")
        std = torch.ones(in_channels)
    else:
        std = torch.sqrt(m2 / count)

    std[std == 0] = 1e-8

    print("✓ Z-score stats calculated.")
    return {'mean': mean.numpy(), 'std': std.numpy()}


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Ghi đè cấu hình từ tham số dòng lệnh
    if args.run_name:
        config['run_name'] = args.run_name
    if args.model_name:
        config['model']['name'] = args.model_name

    # Đảm bảo tồn tại model.params
    model_params = config.setdefault('model', {}).setdefault('params', {})

    if args.lr:
        config['optimizer']['params']['lr'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs

    # Overrides for Loss alpha/beta
    if args.alpha is not None:
        config['loss']['params']['alpha'] = args.alpha
    if args.beta is not None:
        config['loss']['params']['beta'] = args.beta

    # Xử lý tham số --sea (SE Attention)
    if args.sea is not None:
        sac = model_params.get('sac_params')
        if isinstance(sac, dict):
            sac['use_sea'] = bool(args.sea)
            model_params['sac_params'] = sac
            print(f"✓ Set SE Attention (SEA) in SAC module to: {args.sea}")
        else:
            print("Warning: --sea flag has no effect as model does not have 'sac_params'.")

    # Xử lý tham số --channels
    if args.channels:
        # Ưu tiên cập nhật 'selected_channels' nếu có
        if 'selected_channels' in model_params:
            model_params['selected_channels'] = int(args.channels)
            print(f"✓ Set selected_channels to: {args.channels}")
        elif 'in_channels' in model_params:
            model_params['in_channels'] = int(args.channels)
            print(f"✓ Set in_channels to: {args.channels}")
        else:
            print("Warning: '--channels' override failed. Model params lack 'selected_channels' or 'in_channels'.")
    
    # --- Logic mới để xử lý chọn băng tần động ---
    band_rankings = load_band_rankings()
    num_select_bands = model_params.get('in_channels') or model_params.get('selected_channels')
    model_type = config['model']['name']

    # CRITICAL FIX FOR TABS: TABS needs access to all 25 bands to learn selection.
    # If we are running TABS, we should NOT slice the dataset based on rankings.
    # We should sets 'num_selected_bands' instead.
    if model_type in ['TABS', 'ASRAN_LBS', 'UnetBandS']:
        print(f"✓ Detected Learnable Band Selection model ({model_type}).")
        # Ensure input is full 25 bands
        model_params['in_channels'] = 25
        # If args.channels was passed, it likely overwrote in_channels earlier.
        # We interpreted args.channels as 'k' (num_selected_bands) for TABS.
        if args.channels:
            model_params['num_selected_bands'] = int(args.channels)
            print(f"✓ Set num_selected_bands to {args.channels} for TABS.")
        
        # Explicitly remove any static selection indices to prevent Dataset slicing
        if 'selected_bands_indices' in model_params:
            del model_params['selected_bands_indices']

    elif num_select_bands is not None and num_select_bands < 25: # Only select if less than full bands (FOR BASELINES)
        selected_method_ranking = None
        # Use the specified ranking method from the config, with 'SVM' as a fallback.
        ranking_key = config['model'].get('ranking_method', 'SVM')
        if ranking_key in band_rankings:
            selected_method_ranking = band_rankings[ranking_key]
            print(f"✓ Model '{model_type}' using '{ranking_key}' ranking for band selection.")
        else:
            print(f"Warning: Ranking method '{ranking_key}' not found in band_rankings.txt. Falling back to SVM.")
            selected_method_ranking = band_rankings.get("SVM")
        
        if selected_method_ranking:
            model_params['selected_bands_indices'] = selected_method_ranking[:num_select_bands]
            print(f"✓ Automatically selected top {num_select_bands} bands for model '{model_type}' using ranking.")
            print(f"   Selected indices: {model_params['selected_bands_indices']}")
        else:
            print(f"Warning: Could not find suitable band ranking for model '{model_type}'. Using all 25 bands or default in_channels.")
            # Đảm bảo in_channels được đặt là 25 nếu không chọn băng tần
            if 'in_channels' in model_params: model_params['in_channels'] = 25


    run_path, is_new_run = ExperimentManager.setup_run_directory(
        base_output_dir=config['base_output_dir'],
        run_name=config['run_name'],
        resume_path=args.resume_path
    )
    manager = ExperimentManager(run_path, config['model']['class_names'], is_new_run)
    manager.save_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config['data']
    data_dir = Path(data_cfg['dir'])

    # Cho phép chỉ định thư mục con chứa dữ liệu linh hoạt hơn
    cube_dir_name = data_cfg.get('cube_dir_name', 'cubes_fl32')
    cube_dir = data_dir / cube_dir_name

    all_files = sorted(glob.glob(os.path.join(cube_dir, f"*.{data_cfg.get('file_extension', 'mat')}")))
    if not all_files:
        raise FileNotFoundError(f"No .{data_cfg.get('file_extension', 'mat')} files found in {cube_dir}.")

    random.seed(data_cfg.get('seed', 42))
    random.shuffle(all_files)

    # Tách tập dữ liệu train/val/test
    val_split = data_cfg.get('val_split', 0.2)
    test_split = data_cfg.get('test_split', 0.0) # Default to 0.0 (no test set) if not specified

    n_total = len(all_files)
    n_test = int(test_split * n_total)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val - n_test

    # Files distribution: [Train ... | Val ... | Test ...]
    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]

    if test_split > 0:
        print(f"✓ Data Split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test ({test_split*100:.0f}%)")
    else:
        print(f"✓ Data Split: {len(train_files)} Train, {len(val_files)} Val (No Test Set)")

    stats = None
    if config['data'].get('normalization') == 'z-score':
        # Lấy số kênh đầu vào một cách an toàn từ model.params
        # Truyền selected_bands cho calculate_zscore_stats
        in_channels_for_stats = model_params.get('in_channels', model_params.get('selected_channels'))
        if in_channels_for_stats is None:
            raise ValueError("Cannot determine number of input channels for Z-score calculation.")
        
        selected_bands_for_stats = model_params.get('selected_bands_indices', None)
        stats = calculate_zscore_stats(train_files, int(in_channels_for_stats), selected_bands_for_stats)


    train_dataset = TiledHyperspectralDataset(file_paths=train_files, config=config, augment=True, stats=stats)
    val_dataset = TiledHyperspectralDataset(file_paths=val_files, config=config, augment=False, stats=stats)

    num_workers = args.num_workers if args.num_workers is not None else (min(os.cpu_count(), 4) if platform.system() != 'Windows' else 0)

    test_loader = None
    if len(test_files) > 0:
        test_dataset = TiledHyperspectralDataset(file_paths=test_files, config=config, augment=False, stats=stats)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            print(f"✓ Test loader ready with {len(test_dataset)} patches.")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training or validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"✓ Data ready: {len(train_dataset)} train patches, {len(val_dataset)} validation patches.")

    # Tạo model
    model = get_model(config['model']['name'], config['model']).to(device)

    # This now needs to iterate through the dataset to get masks due to lazy loading
    # --- Caching Mechanism for Class Weights ---
    metadata_dir = "dataset_metadata"
    os.makedirs(metadata_dir, exist_ok=True)
    seed_val = config['data'].get('seed', 42)
    weights_cache_file = os.path.join(metadata_dir, f"class_weights_seed_{seed_val}.npy")

    if os.path.exists(weights_cache_file):
        print(f"✓ Loading cached class weights from: {weights_cache_file}")
        class_weights = np.load(weights_cache_file)
    else:
        print("   Calculating class weights from training patches (this may take a while)...")
        # Chỉ lấy mask (index 1) để tiết kiệm memory/thời gian nếu dataset hỗ trợ
        # Tuy nhiên dataset hiện tại trả về tuple (img, mask), ta cần duyệt qua
        all_masks_pixels = np.concatenate([
            train_dataset[i][1].numpy().flatten() for i in tqdm(range(len(train_dataset)), desc="Sampling weights")
        ])
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(config['model']['params']['num_classes']),
            y=all_masks_pixels
        )
        np.save(weights_cache_file, class_weights)
        print(f"✓ Calculated and cached class weights to: {weights_cache_file}")
    
    print(f"✓ Class weights: {class_weights}")

    loss_cfg = config.get('loss') or config['training'].get('loss') or config['training'].get('loss_function')
    if not loss_cfg:
        raise ValueError("Loss function configuration not found in config['training']")

    criterion = get_loss(
        name=loss_cfg['name'],
        params=loss_cfg.get('params', {}),
        class_weights=class_weights,
        device=device
    ).to(device)

    optimizer_cfg = config.get('optimizer') or config['training']['optimizer']
    optimizer = getattr(optim, optimizer_cfg['name'])(model.parameters(), **optimizer_cfg['params'])

    # Cấu hình scheduler linh hoạt hơn
    scheduler_cfg = config.get('scheduler') or config['training'].get('scheduler')
    if not scheduler_cfg:
        raise ValueError("Scheduler configuration not found in config['training']")
    scheduler_params = scheduler_cfg.get('params', {}).copy()
    scheduler_name = scheduler_cfg['name']
    if scheduler_name == 'CosineAnnealingLR':
        scheduler_params['T_max'] = config['training']['num_epochs']

    scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_params)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        train_loader=train_loader, val_loader=val_loader,
        manager=manager, device=device, config=config,
        num_epochs=config['training'].get('epochs', config['training']['num_epochs'])
    )
    trainer.train()

    # --- Final Test Set Evaluation ---
    if test_loader is not None:
        trainer.evaluate_test_set(test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Unified Training Script for Segmentation Models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/test_config.yaml', help='Path to the base configuration file.')
    parser.add_argument('--run_name', type=str, default=None, help='Override the run name from the config file.')
    parser.add_argument('--model_name', type=str, default=None, help='Override the model name from the config file.')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to a specific run directory to resume.')
    parser.add_argument('--channels', type=int, default=None, help='Override number of input/selected channels.')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate.')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size.')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs.')
    parser.add_argument('--alpha', type=float, default=None, help='Override alpha for CombinedLoss.')
    parser.add_argument('--beta', type=float, default=None, help='Override beta for CombinedLoss.')
    parser.add_argument('--sea', type=lambda x: (str(x).lower() == 'true'), default=None,
                        help='Enable or disable SE Attention in SAC module (True/False).')
    parser.add_argument('--num_workers', type=int, default=None, help='Override number of data loading workers.')

    args = parser.parse_args()
    main(args)
