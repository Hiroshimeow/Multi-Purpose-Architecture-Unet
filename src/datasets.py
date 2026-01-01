# src/datasets.py
import os
import numpy as np
import cv2
from PIL import Image
import h5py
import scipy.io as io
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class TiledHyperspectralDataset(Dataset):
    def __init__(self, file_paths: list, config: dict, augment=False, stats=None):
        self.config = config
        self.augment = augment
        self.file_paths_raw = file_paths # Keep original file paths for __getitem__
        self.stats = stats

        # Support for specific band selection
        self.selected_bands = config['model']['params'].get('selected_bands_indices', None)
        self.num_selected_bands = len(self.selected_bands) if self.selected_bands is not None else 25

        if self.selected_bands is not None:
             print(f"Dataset: Selecting specific bands indices: {self.selected_bands}")
        else:
             print(f"Dataset: Using all {self.num_selected_bands} available bands.")


        ps = config['data']['patching']['patch_size']
        st = config['data']['patching']['stride']
        self.patch_size = (ps, ps) if isinstance(ps, int) else tuple(ps)
        self.stride = (st, st) if isinstance(st, int) else tuple(st)
        
        self.class_mapping = config['data']['class_mapping']
        self.num_classes = config['model']['params']['num_classes']

        aug_level = config['data'].get('augmentation_level', 'default')
        
        transforms_list = []
        if aug_level == 'paper':
            print("Using 'paper' augmentation pipeline.")
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2)
            ])
        else: # Default, backward-compatible behavior
            print("Using 'default' configurable augmentation pipeline.")
            aug_cfg = config.get('augmentation', {})
            # ... (logic cũ giữ nguyên)
            transforms_list.extend([A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5)])

        transforms_list.append(ToTensorV2())
        self.transforms = A.Compose(transforms_list)
        self.val_transforms = A.Compose([ToTensorV2()])

        # Instead of storing actual patches, store metadata (file_path, coords)
        self.patch_metadata = [] 
        print(f"Collecting patch coordinates for {len(self.file_paths_raw)} files for {'training' if augment else 'validation'} set...")
        for cube_path in tqdm(self.file_paths_raw):
            self._collect_patch_coordinates(cube_path)
        
        if not self.patch_metadata:
            print(f"WARNING: No patch metadata were generated for this dataset part.")

    def _normalize_cube(self, cube):
        if self.config['data'].get('normalization') == 'z-score':
            if self.stats is None: raise ValueError("Z-score normalization requires stats (mean, std).")
            mean, std = self.stats['mean'], self.stats['std']
            
            # If selected_bands are used, stats also need to be for selected bands
            if self.selected_bands is not None:
                # Assuming stats are already calculated for the selected bands in calculate_zscore_stats
                # Or, if stats are calculated on full 25 bands, we need to slice them here.
                # For consistency, calculate_zscore_stats is modified to use selected_bands
                pass 

            # Reshape for broadcasting (1, 1, C)
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)

            return (cube - mean) / (std + 1e-8)
        else: # Per-image min-max
            # Operate on the num_selected_bands (or 25 if no selection)
            for i in range(cube.shape[2]): 
                channel = cube[:, :, i]
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val > min_val: cube[:, :, i] = (channel - min_val) / (max_val - min_val)
            return cube
        
    def _pad_if_needed(self, image, mask):
        h, w, _ = image.shape
        ph, pw = self.patch_size
        pad_h, pad_w = max(0, ph - h), max(0, pw - w)
        if pad_h > 0 or pad_w > 0:
            pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
            pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return image, mask

    def _collect_patch_coordinates(self, cube_file_path):
        base_name = os.path.splitext(os.path.basename(cube_file_path))[0]
        mask_path_dir = os.path.dirname(cube_file_path).replace('cubes_fl32', 'labels')
        mask_file_path = os.path.join(mask_path_dir, f"{base_name}.png")
        if not os.path.exists(mask_file_path): mask_file_path = os.path.join(mask_path_dir, f"{base_name.replace('_TC', '')}.png")

        try:
            # We only load a small part of the cube to get its shape for tiling calculation
            # This is efficient as we don't load the full data into memory yet
            _dummy_cube = self._load_cube_header(cube_file_path) # New helper to load only shape
            _dummy_mask = self._load_mask(mask_file_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Warning: Skipping file {cube_file_path} due to error: {e}")
            return
        
        # Apply band selection to dummy cube to get correct shape for tiling
        if self.selected_bands is not None:
             _dummy_cube = _dummy_cube[:, :, self.selected_bands]
        
        _dummy_cube, _dummy_mask = self._pad_if_needed(_dummy_cube, _dummy_mask) # Pad dummy data for tiling

        h, w, _ = _dummy_cube.shape
        ph, pw = self.patch_size
        sh, sw = self.stride

        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                self.patch_metadata.append({
                    'cube_path': cube_file_path,
                    'mask_path': mask_file_path,
                    'y': y, 'x': x,
                    'ph': ph, 'pw': pw
                })

    def __len__(self):
        return len(self.patch_metadata)

    def __getitem__(self, idx):
        metadata = self.patch_metadata[idx]
        cube_path = metadata['cube_path']
        mask_path = metadata['mask_path']
        y, x = metadata['y'], metadata['x']
        ph, pw = metadata['ph'], metadata['pw']

        # Load full cube and mask on demand
        full_cube = self._load_cube(cube_path)
        full_mask = self._load_mask(mask_path)
        
        # Apply band selection if configured
        if self.selected_bands is not None:
            full_cube = full_cube[:, :, self.selected_bands]
        
        full_mask = self._apply_class_mapping(full_mask)
        full_cube = self._normalize_cube(full_cube)
        
        # Pad again if needed, this time to the actual image to avoid boundary issues during tiling
        full_cube, full_mask = self._pad_if_needed(full_cube, full_mask)

        cube_patch = full_cube[y:y+ph, x:x+pw, :]
        mask_patch = full_mask[y:y+ph, x:x+pw]

        transforms = self.transforms if self.augment else self.val_transforms
        transformed = transforms(image=cube_patch, mask=mask_patch)
        
        t_mask = transformed['mask']

        # DEBUGGING: Check for out-of-bounds mask values
        if torch.any(t_mask >= self.num_classes) or torch.any(t_mask < 0):
            unique_vals = torch.unique(t_mask)
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!! INVALID MASK VALUES DETECTED: {unique_vals}")
            print(f"!!! MASK SHOULD BE IN RANGE [0, {self.num_classes - 1}]")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        return transformed['image'], t_mask.long()

    # Helper to load only shape for tiling calculation
    def _load_cube_header(self, cube_path):
        try:
            # Use io.loadmat which is more robust to different .mat file versions
            # Although it loads the whole file, it's necessary for stability here.
            mat_data = io.loadmat(cube_path, mat_dtype=True, variable_names=[k[0] for k in io.whosmat(cube_path) if not k[0].startswith('__')])
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not data_keys: raise ValueError(f"No data found in {cube_path}")
            cube_shape = mat_data[data_keys[0]].shape

            if len(cube_shape) != 3:
                raise ValueError(f"Cube {cube_path} is not 3D, shape is {cube_shape}")

            # Find the channel dimension and return a dummy array of the transposed shape
            try:
                channel_dim_index = cube_shape.index(25)
            except ValueError:
                raise ValueError(f"Cube {cube_path} has shape {cube_shape}, but no dimension of size 25 was found.")

            if channel_dim_index == 0: # (C, H, W)
                return np.zeros((cube_shape[1], cube_shape[2], cube_shape[0]), dtype=np.float32)
            elif channel_dim_index == 1: # (H, C, W)
                return np.zeros((cube_shape[0], cube_shape[2], cube_shape[1]), dtype=np.float32)
            else: # (H, W, C)
                return np.zeros(cube_shape, dtype=np.float32)

        except Exception as e:
            # print(f"Error reading header for {cube_path}: {e}")
            raise
        
    def _load_cube(self, cube_path):
        ext = os.path.splitext(cube_path)[1]
        if ext == '.mat':
            mat_data = io.loadmat(cube_path)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not data_keys: raise ValueError(f"No data found in {cube_path}")
            cube = mat_data[data_keys[0]].astype(np.float32)
        elif ext == '.h5':
            with h5py.File(cube_path, 'r') as f:
                key = list(f.keys())[0]
                cube = np.array(f[key]).astype(np.float32)
        else:
            raise ValueError(f"Unsupported cube file format: {ext}")

        # New, simpler transpose logic
        if cube.ndim != 3:
            raise ValueError(f"Cube {cube_path} has {cube.ndim} dimensions, expected 3.")

        # Find the channel dimension (should be 25 for the full cube, or k for sliced cubes)
        # This logic assumes the full cube is always loaded first from disk.
        try:
            channel_dim_index = cube.shape.index(25)
        except ValueError:
            # This path should ideally not be taken if we always load full cubes.
            # But as a fallback, we can check for the selected number of bands.
            try:
                channel_dim_index = cube.shape.index(self.num_selected_bands)
            except ValueError:
                raise ValueError(f"Cube {cube_path} has shape {cube.shape}, but neither a dimension of 25 nor {self.num_selected_bands} was found.")

        # Permute to (H, W, C)
        if channel_dim_index == 0: # (C, H, W) -> (H, W, C)
            cube = np.transpose(cube, (1, 2, 0))
        elif channel_dim_index == 1: # (H, C, W) -> (H, W, C)
            cube = np.transpose(cube, (0, 2, 1))
        # if channel_dim_index is 2, it's already (H, W, C), do nothing.

        return cube

    def _load_mask(self, mask_path):
        return np.array(Image.open(mask_path)).astype(np.uint8)

    def _apply_class_mapping(self, mask):
        mapped_mask = np.zeros_like(mask)
        for original_class_id, new_class_id in self.class_mapping.items():
            mapped_mask[mask == original_class_id] = new_class_id
        return mapped_mask