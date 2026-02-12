import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from autoencoder import HyperspectralAutoencoder
import yaml
from pathlib import Path
import shutil
import scipy.io as io
import h5py
import cv2
from PIL import Image

def _normalize_cube(cube, config):
    # Using min-max normalization as in autoencoder training
    for i in range(cube.shape[2]):
        channel = cube[:, :, i]
        min_val, max_val = np.min(channel), np.max(channel)
        if max_val > min_val:
            cube[:, :, i] = (channel - min_val) / (max_val - min_val)
    return cube

def _load_cube(cube_path, in_channels):
    try:
        mat_data = io.loadmat(cube_path)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if not data_keys: raise ValueError(f"No data found in {cube_path}")
        cube = mat_data[data_keys[0]].astype(np.float32)
        if cube.shape[0] == in_channels: cube = np.transpose(cube, (1, 2, 0))
    except Exception:
        with h5py.File(cube_path, 'r') as f:
            key = list(f.keys())[0]
            cube = np.array(f[key]).astype(np.float32)
            if cube.shape[0] == in_channels: cube = np.transpose(cube, (1, 2, 0))
    return cube

def generate_3band_dataset(config, encoder_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained encoder
    encoder = HyperspectralAutoencoder(
        in_channels=config['model']['params']['in_channels'],
        latent_dim=3
    ).encoder
    encoder.load_state_dict(torch.load(encoder_path))
    encoder = encoder.to(device)
    encoder.eval()

    # Create output directories
    output_cubes_dir = os.path.join(output_dir, 'cubes_fl32')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_cubes_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Get all file paths from the original dataset
    data_cfg = config['data']
    data_dir = Path(data_cfg['dir'])
    cube_dir = data_dir / "cubes_fl32"
    all_files = sorted(glob.glob(os.path.join(cube_dir, "*.mat")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(cube_dir, "*.h5")))

    print(f"Found {len(all_files)} files to process.")

    with torch.no_grad():
        for cube_path in tqdm(all_files, desc="Generating 3-band dataset"):
            base_name = os.path.splitext(os.path.basename(cube_path))[0]
            
            # Load and normalize the full cube
            full_cube = _load_cube(cube_path, config['model']['params']['in_channels'])
            normalized_cube = _normalize_cube(full_cube.copy(), config)
            
            # Convert to tensor and add batch dimension
            cube_tensor = torch.from_numpy(normalized_cube).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Get the 3-band representation from the encoder
            encoded_cube = encoder(cube_tensor)
            
            # Save the new 3-band image as a .mat file
            encoded_cube_np = encoded_cube.squeeze(0).cpu().numpy()
            output_mat_path = os.path.join(output_cubes_dir, f"{base_name}.mat")
            io.savemat(output_mat_path, {'data': encoded_cube_np})

            # Copy the corresponding label file
            mask_path_dir = os.path.dirname(cube_path).replace('cubes_fl32', 'labels')
            mask_path = os.path.join(mask_path_dir, f"{base_name}.png")
            if not os.path.exists(mask_path):
                 mask_path = os.path.join(mask_path_dir, f"{base_name.replace('_TC', '')}.png")
            
            if os.path.exists(mask_path):
                shutil.copy(mask_path, os.path.join(output_labels_dir, f"{base_name}.png"))
            else:
                print(f"Warning: Mask not found for {cube_path}")

    print(f"Successfully generated 3-band dataset in {output_dir}")

if __name__ == '__main__':
    with open('configs/run9_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generate_3band_dataset(
        config=config,
        encoder_path='encoder.pth',
        output_dir='Image_dataset_3band'
    )
