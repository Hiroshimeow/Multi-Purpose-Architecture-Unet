import numpy as np
import os
import glob
from tqdm import tqdm
import yaml
from pathlib import Path
import shutil
import scipy.io as io
import h5py
from sklearn.decomposition import PCA
import joblib

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

def generate_pca_dataset(config, output_dir, n_components=3):
    data_cfg = config['data']
    data_dir = Path(data_cfg['dir'])
    cube_dir = data_dir / "cubes_fl32"
    all_files = sorted(glob.glob(os.path.join(cube_dir, "*.mat")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(cube_dir, "*.h5")))

    print(f"Found {len(all_files)} files to process.")

    # Fit PCA on a subset of the data
    print("Fitting PCA model...")
    # Taking a small sample to fit PCA
    sample_files = np.random.choice(all_files, size=min(len(all_files), 50), replace=False)
    sample_data = []
    for cube_path in tqdm(sample_files, desc="Loading sample data for PCA"):
        cube = _load_cube(cube_path, config['model']['params']['in_channels'])
        h, w, c = cube.shape
        sample_data.append(cube.reshape(h * w, c))
    
    sample_data = np.vstack(sample_data)
    
    pca = PCA(n_components=n_components)
    pca.fit(sample_data)
    
    # Save the PCA model
    pca_model_path = os.path.join(output_dir, 'pca_model.joblib')
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pca, pca_model_path)
    print(f"PCA model saved to {pca_model_path}")

    # Create output directories
    output_cubes_dir = os.path.join(output_dir, 'cubes_fl32')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_cubes_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Transform and save the entire dataset
    for cube_path in tqdm(all_files, desc="Generating PCA 3-band dataset"):
        base_name = os.path.splitext(os.path.basename(cube_path))[0]
        
        full_cube = _load_cube(cube_path, config['model']['params']['in_channels'])
        h, w, c = full_cube.shape
        
        # Transform the cube
        transformed_cube = pca.transform(full_cube.reshape(h * w, c))
        transformed_cube = transformed_cube.reshape(h, w, n_components)

        # Save the new 3-band image as a .mat file
        output_mat_path = os.path.join(output_cubes_dir, f"{base_name}.mat")
        io.savemat(output_mat_path, {'data': transformed_cube.astype(np.float32)})

        # Copy the corresponding label file
        mask_path_dir = os.path.dirname(cube_path).replace('cubes_fl32', 'labels')
        mask_path = os.path.join(mask_path_dir, f"{base_name}.png")
        if not os.path.exists(mask_path):
             mask_path = os.path.join(mask_path_dir, f"{base_name.replace('_TC', '')}.png")
        
        if os.path.exists(mask_path):
            shutil.copy(mask_path, os.path.join(output_labels_dir, f"{base_name}.png"))
        else:
            print(f"Warning: Mask not found for {cube_path}")

    print(f"Successfully generated PCA 3-band dataset in {output_dir}")

if __name__ == '__main__':
    with open('configs/run12_pca_5band_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generate_pca_dataset(
        config=config,
        output_dir='Image_dataset_pca_5band',
        n_components=5
    )
