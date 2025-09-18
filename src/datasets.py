# src/datasets.py
import os
import numpy as np
import cv2
from PIL import Image
import h5py
import scipy.io as io
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from src.utils import generate_boundary_weight_map

class TiledHyperspectralDataset(Dataset):
    def __init__(self, file_paths: list, config: dict, augment=False, stats=None):
        self.config = config
        self.augment = augment
        self.file_paths = file_paths
        self.stats = stats

        ps = config['data']['patching']['patch_size']
        st = config['data']['patching']['stride']
        self.patch_size = (ps, ps) if isinstance(ps, int) else tuple(ps)
        self.stride = (st, st) if isinstance(st, int) else tuple(st)
        
        aug_cfg = config.get('augmentation', {})
        elastic_params = aug_cfg.get('elastic_transform', {'p': 0.0})
        ssr_params = aug_cfg.get('shift_scale_rotate', {'p': 0.0})
        rbc_params = aug_cfg.get('random_brightness_contrast', {'p': 0.0})
        gn_params = aug_cfg.get('gauss_noise', {'p': 0.0})

        self.class_mapping = config['data']['class_mapping']
        self.in_channels = config['model']['params']['in_channels']
        
        transforms_list = [A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5)]
        if elastic_params['p'] > 0: transforms_list.append(A.ElasticTransform(p=elastic_params['p'], alpha=elastic_params['alpha'], sigma=elastic_params['sigma'], border_mode=cv2.BORDER_REFLECT_101))
        if ssr_params['p'] > 0: transforms_list.append(A.ShiftScaleRotate(p=ssr_params['p'], shift_limit=ssr_params['shift_limit'], scale_limit=ssr_params['scale_limit'], rotate_limit=ssr_params['rotate_limit'], border_mode=cv2.BORDER_REFLECT_101))
        if rbc_params['p'] > 0: transforms_list.append(A.RandomBrightnessContrast(p=rbc_params['p']))
        if gn_params['p'] > 0: transforms_list.append(A.GaussNoise(p=gn_params['p']))
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
            if self.stats is None:
                raise ValueError("Z-score normalization requires stats (mean, std).")
            mean, std = self.stats['mean'], self.stats['std']
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)
            return (cube - mean) / (std + 1e-8)
        else:
            for i in range(cube.shape[2]):
                channel = cube[:, :, i]
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val > min_val:
                    cube[:, :, i] = (channel - min_val) / (max_val - min_val)
            return cube
        
    def _pad_if_needed(self, image, mask):
        h, w, _ = image.shape
        ph, pw = self.patch_size
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
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
        if not os.path.exists(mask_path):
             mask_path = os.path.join(mask_path_dir, f"{base_name.replace('_TC', '')}.png")

        try:
            full_cube = self._load_cube(cube_path)
            full_mask = self._load_mask(mask_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Skipping file due to error: {e}")
            return

        full_mask = self._apply_class_mapping(full_mask)
        full_cube = self._normalize_cube(full_cube)
        full_cube, full_mask = self._pad_if_needed(full_cube, full_mask)
        
        h, w, _ = full_cube.shape
        ph, pw = self.patch_size
        sh, sw = self.stride

        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                cube_patch = full_cube[y:y+ph, x:x+pw, :]
                mask_patch = full_mask[y:y+ph, x:x+pw]
                
                if np.any(mask_patch > 0):
                    # FIX: Không tạo weight map nữa. Chỉ lưu (ảnh, mask).
                    self.patches.append((cube_patch, mask_patch))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # FIX: Chỉ lấy cube và mask từ self.patches
        cube_patch, mask_patch = self.patches[idx]
        
        transforms = self.transforms if self.augment else self.val_transforms
        
        # FIX: Áp dụng transform chỉ cho image và mask
        transformed = transforms(image=cube_patch, mask=mask_patch)
        
        image = transformed['image']
        t_mask = transformed['mask']

        # FIX: Chỉ trả về image và mask
        return image, t_mask.long()

    def _load_cube(self, cube_path):
        try:
            mat_data = io.loadmat(cube_path)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not data_keys: raise ValueError(f"No data found in {cube_path}")
            cube = mat_data[data_keys[0]].astype(np.float32)
            # Ensure cube is (H, W, C)
            if cube.ndim == 3 and cube.shape[0] < cube.shape[2] and cube.shape[0] < cube.shape[1]: # Assume C, H, W
                cube = np.transpose(cube, (1, 2, 0)) # Convert to H, W, C
        except Exception:
            with h5py.File(cube_path, 'r') as f:
                key = list(f.keys())[0]
                cube = np.array(f[key]).astype(np.float32)
                # Ensure cube is (H, W, C)
                if cube.ndim == 3 and cube.shape[0] < cube.shape[2] and cube.shape[0] < cube.shape[1]: # Assume C, H, W
                    cube = np.transpose(cube, (1, 2, 0)) # Convert to H, W, C
        return cube

    def _load_mask(self, mask_path):
        return np.array(Image.open(mask_path)).astype(np.uint8)

    def _apply_class_mapping(self, mask):
        mapped_mask = np.zeros_like(mask)
        for original_class, new_class in self.class_mapping.items():
            mapped_mask[mask == original_class] = new_class
        return mapped_mask