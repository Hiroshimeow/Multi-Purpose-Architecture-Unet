# run.py
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
# FIX: Import get_loss thay vì các class cụ thể
from src.losses import get_loss
from src.experiment_manager import ExperimentManager
from src.trainer import Trainer

def acquire_lock(lock_file_path: Path):
    lock_file = lock_file_path
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            print(f"Acquired lock: {lock_file}")
            return
        except FileExistsError:
            print("Another process is initializing. Waiting 30 seconds...")
            time.sleep(30)

def release_lock(lock_file_path: Path):
    try:
        if lock_file_path.exists():
            os.remove(lock_file_path)
            print(f"Released lock: {lock_file_path}")
    except OSError as e:
        print(f"Error releasing lock file: {e}")
def calculate_zscore_stats(file_paths: list, in_channels: int):
    """
    Calculates mean and standard deviation across the training set for Z-score normalization.
    This is crucial for ensuring the model receives consistently scaled data.
    """
    # FIX: Import các thư viện cần thiết vào scope của hàm
    from torch.utils.data import Dataset
    import torch
    # Không cần import h5py và io ở đây nữa vì chúng đã có ở scope toàn cục

    print("Calculating Z-score statistics on the training set...")
    
    class StatsDataset(Dataset):
        def __init__(self, files, channels):
            self.files = files
            self.channels = channels
        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            # Logic đọc file đã được cải thiện
            try:
                with h5py.File(self.files[idx], 'r') as f:
                    data_keys = list(f.keys())
                    if not data_keys: return None
                    cube = np.array(f[data_keys[0]]).astype(np.float32)
                    if cube.shape[0] == self.channels and cube.ndim == 3:
                        cube = np.transpose(cube, (1, 2, 0))
                    return torch.from_numpy(cube)
            except OSError:
                try:
                    mat_data = io.loadmat(self.files[idx])
                    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    if not data_keys: return None
                    cube = mat_data[data_keys[0]].astype(np.float32)
                    if cube.shape[0] == self.channels and cube.ndim == 3:
                        cube = np.transpose(cube, (1, 2, 0))
                    return torch.from_numpy(cube)
                except Exception as e:
                    print(f"Warning: Could not load {self.files[idx]} with scipy. Error: {e}")
                    return None
            except Exception as e:
                print(f"Warning: An unexpected error occurred with {self.files[idx]}. Error: {e}")
                return None

    dataset = StatsDataset(file_paths, in_channels)
    
    # FIX: Thêm collate_fn để xử lý các giá trị None một cách an toàn
    def collate_fn_skip_none(batch):
        # Lọc ra các item không phải là None
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Trả về None nếu cả batch đều lỗi
        # Sử dụng collate mặc định cho batch đã được lọc
        return torch.utils.data.default_collate(batch)

    loader = DataLoader(dataset, batch_size=4, num_workers=platform.system() != 'Windows' and 4 or 0, shuffle=False, collate_fn=collate_fn_skip_none)
    
    count = 0
    mean = torch.zeros(in_channels)
    m2 = torch.zeros(in_channels)

    for cube_batch in tqdm(loader, desc="Calculating Stats"):
        # Nếu collate_fn trả về None (cả batch lỗi), bỏ qua
        if cube_batch is None: continue
        
        # DataLoader trả về một batch tensor, không cần duyệt qua từng cube
        pixels = cube_batch.view(-1, in_channels)
        n = len(pixels)
        if n == 0: continue
        
        batch_mean = pixels.mean(dim=0)
        batch_var = pixels.var(dim=0, unbiased=False) # Population variance for the batch
        
        delta = batch_mean - mean
        mean += delta * n / (count + n)
        m2 += batch_var * n + delta**2 * count * n / (count + n)
        count += n

    if count < 2:
        print("Warning: Not enough data to calculate standard deviation. Returning std=1.")
        std = torch.ones(in_channels)
    else:
        # Population variance is m2 / count
        std = torch.sqrt(m2 / count)
    
    # Thêm một epsilon nhỏ để tránh std = 0
    std[std == 0] = 1e-8
    
    print("✓ Z-score stats calculated.")
    return {'mean': mean.numpy(), 'std': std.numpy()}

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.run_name: config['run_name'] = args.run_name
    if args.lr: config['optimizer']['params']['lr'] = args.lr
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.channels: 
        if 'selected_channels' in config['model']['params']:
            config['model']['params']['selected_channels'] = args.channels
        else:
            print("Warning: '--channels' override is only supported for models with 'selected_channels' param (e.g., HybridUNet).")

    run_path, is_new_run = ExperimentManager.setup_run_directory(
        base_output_dir=config['base_output_dir'],
        run_name=config['run_name'],
        resume_path=args.resume_path
    )
    manager = ExperimentManager(run_path, config['class_names'], is_new_run)
    manager.save_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config['data']
    data_dir = Path(data_cfg['dir'])
    cube_dir = data_dir / "cubes_fl32"
    
    all_files = sorted(glob.glob(os.path.join(cube_dir, "*.mat")))
    if not all_files:
        raise FileNotFoundError(f"No .mat files found in {cube_dir}.")
        
    random.seed(42)
    random.shuffle(all_files)
    
    split_idx = int(0.8 * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    
    stats = None
    if config['data'].get('normalization') == 'z-score':
        # FIX: Sửa lại logic get in_channels cho an toàn
        c_param = 'selected_channels' if 'selected_channels' in config['model']['params'] else 'in_channels'
        in_channels = config['model']['params'][c_param]
        stats = calculate_zscore_stats(train_files, in_channels)
    
    train_dataset = TiledHyperspectralDataset(file_paths=train_files, config=config, augment=True, stats=stats)
    val_dataset = TiledHyperspectralDataset(file_paths=val_files, config=config, augment=False, stats=stats)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training or validation dataset is empty.")

    num_workers = min(os.cpu_count(), 8) if platform.system() != 'Windows' else 0
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"✓ Data ready: {len(train_dataset)} train patches, {len(val_dataset)} validation patches.")

    model = get_model(config['model']['name'], config['model']['params']).to(device)
    
    print("   Calculating class weights from training patches...")
    # NOTE: Dữ liệu từ dataset giờ chỉ là (ảnh, mask)
    all_masks_pixels = np.concatenate([p[1].flatten() for p in tqdm(train_dataset.patches, desc="Sampling weights")])
    class_weights = compute_class_weight('balanced', classes=np.arange(config['model']['params']['num_classes']), y=all_masks_pixels)
    print(f"✓ Calculated class weights: {class_weights}")
    
    # FIX: Sử dụng loss factory để khởi tạo criterion
    criterion = get_loss(
        name=config['loss']['name'],
        params=config['loss'].get('params', {}),
        class_weights=class_weights
    ).to(device)
    
    optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), **config['optimizer']['params'])
    
    scheduler_params = config['scheduler']['params'].copy() # Dùng copy để tránh pop làm thay đổi dict gốc
    if config['scheduler']['name'] == 'CosineAnnealingLR':
        scheduler_params['T_max'] = config['training']['num_epochs'] + scheduler_params.pop('T_max_epochs_offset', 0)
    scheduler = getattr(optim.lr_scheduler, config['scheduler']['name'])(optimizer, **scheduler_params)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        train_loader=train_loader, val_loader=val_loader,
        manager=manager, device=device, config=config
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a Segmentation Model Training Experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/unet_paper_replica.yaml', help='Path to the base configuration file.')
    parser.add_argument('--run_name', type=str, default=None, help='Override the run name from the config file.')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to a specific run directory to resume.')
    parser.add_argument('--channels', type=int, default=None, help='Override number of selected channels.')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate.')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size.')
    
    args = parser.parse_args()
    
    config_for_lock = yaml.safe_load(open(args.config, 'r'))
    lock_file = Path(config_for_lock['base_output_dir']) / 'training.lock'
    
    acquire_lock(lock_file)
    try:
        main(args)
    finally:
        release_lock(lock_file)