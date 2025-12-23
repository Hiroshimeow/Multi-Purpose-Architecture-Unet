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
        self.file_paths = file_paths
        self.stats = stats

        # Support for specific band selection
        self.selected_bands = config['model']['params'].get('selected_bands_indices', None)
        if self.selected_bands is not None:
             print(f"Dataset: Selecting specific bands indices: {self.selected_bands}")

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

        self.patches = []
        print(f"Processing {len(self.file_paths)} files for {'training' if augment else 'validation'} set...")
        for cube_path in tqdm(self.file_paths):
            self._process_and_tile_image(cube_path)
        
        if not self.patches:
            print(f"WARNING: No patches were generated for this dataset part.")

    def _normalize_cube(self, cube):
        if self.config['data'].get('normalization') == 'z-score':
            if self.stats is None: raise ValueError("Z-score normalization requires stats (mean, std).")
            mean, std = self.stats['mean'].reshape(1, 1, -1), self.stats['std'].reshape(1, 1, -1)
            # If we selected bands, we need to slice the stats too if they were calculated on full 25 bands
            # But usually stats are calculated on the input passed to the model.
            # Assuming stats match the cube shape here.
            return (cube - mean) / (std + 1e-8)
        else: # Per-image min-max
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

    def _process_and_tile_image(self, cube_path):
        base_name = os.path.splitext(os.path.basename(cube_path))[0]
        mask_path_dir = os.path.dirname(cube_path).replace('cubes_fl32', 'labels')
        mask_path = os.path.join(mask_path_dir, f"{base_name}.png")
        if not os.path.exists(mask_path): mask_path = os.path.join(mask_path_dir, f"{base_name.replace('_TC', '')}.png")

        try:
            full_cube = self._load_cube(cube_path)
            full_mask = self._load_mask(mask_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            # Catch OSError for bad image files (corrupted during unzip)
            # print(f"Warning: Skipping file due to error: {e}")
            return

        # Slice bands if selected_bands is set
        if self.selected_bands is not None:
            try:
                full_cube = full_cube[:, :, self.selected_bands]
            except IndexError as e:
                 print(f"Error slicing cube {cube_path}: {e}")
                 return

        full_mask = self._apply_class_mapping(full_mask)
        full_cube = self._normalize_cube(full_cube)
        full_cube, full_mask = self._pad_if_needed(full_cube, full_mask)
        
        h, w, _ = full_cube.shape
        ph, pw = self.patch_size
        sh, sw = self.stride

        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                self.patches.append((full_cube[y:y+ph, x:x+pw, :], full_mask[y:y+ph, x:x+pw]))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        cube_patch, mask_patch = self.patches[idx]
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
        
        if cube.ndim == 3 and cube.shape[0] < cube.shape[2] and cube.shape[0] < cube.shape[1]:
            cube = np.transpose(cube, (1, 2, 0))
        return cube

    def _load_mask(self, mask_path):
        return np.array(Image.open(mask_path)).astype(np.uint8)

    def _apply_class_mapping(self, mask):
        mapped_mask = np.zeros_like(mask)
        for original_class_id, new_class_id in self.class_mapping.items():
            mapped_mask[mask == original_class_id] = new_class_id
        return mapped_mask