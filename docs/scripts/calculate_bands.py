import os
import numpy as np
import h5py
import scipy.io as io
from tqdm import tqdm
import glob
import random
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC
import pandas as pd
from PIL import Image

# Config
DATA_DIR = "/data1/tabs/Image_dataset"
CUBE_DIR = os.path.join(DATA_DIR, "cubes_fl32")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
NUM_SAMPLES = 50  # Use 50 images for stats calculation to be fast but representative
SAMPLE_PIXELS = 1000 # Per image per class for SVM

def load_data(cube_path, label_path):
    try:
        # Load cube
        mat = io.loadmat(cube_path)
        key = [k for k in mat.keys() if not k.startswith('__')][0]
        cube = mat[key]
        if cube.ndim == 3 and cube.shape[0] < cube.shape[2]: # Fix HWC if needed
            cube = np.transpose(cube, (1, 2, 0))
            
        # Load label
        label = np.array(Image.open(label_path))
        
        return cube, label
    except Exception as e:
        return None, None

def get_files():
    files = glob.glob(os.path.join(CUBE_DIR, "*.mat"))
    random.shuffle(files)
    return files[:NUM_SAMPLES]

def calculate_rankings():
    files = get_files()
    print(f"Analyzing {len(files)} files...")
    
    # 1. Entropy
    print("Calculating Entropy...")
    entropy_scores = np.zeros(25)
    for f in tqdm(files):
        label_path = f.replace("cubes_fl32", "labels").replace(".mat", ".png").replace("_TC", "")
        if not os.path.exists(label_path): continue
        cube, _ = load_data(f, label_path)
        if cube is None: continue
        
        # Simple entropy of pixel distribution per band
        for b in range(25):
            band = cube[:,:,b].flatten()
            hist, _ = np.histogram(band, bins=256, density=True)
            hist = hist[hist > 0]
            entropy_scores[b] -= np.sum(hist * np.log2(hist))
            
    entropy_ranking = np.argsort(entropy_scores)[::-1] # High entropy first
    
    # 2. Variance (Proxy for information content)
    print("Calculating Variance...")
    var_scores = np.zeros(25)
    for f in tqdm(files):
        cube, _ = load_data(f, f.replace("cubes_fl32", "labels").replace(".mat", ".png").replace("_TC", ""))
        if cube is None: continue
        var_scores += np.var(cube, axis=(0,1))
    
    var_ranking = np.argsort(var_scores)[::-1]

    # 3. SVM-RFE & Fisher & Correlation
    print("Calculating SVM-RFE, Fisher, and Correlation...")
    X_list = []
    y_list = []
    
    for f in tqdm(files[:20]): # Increased to 20 for better stats
        label_path = f.replace("cubes_fl32", "labels").replace(".mat", ".png").replace("_TC", "")
        if not os.path.exists(label_path): continue
        cube, label = load_data(f, label_path)
        if cube is None: continue
        
        # Sample pixels
        h, w = label.shape
        pixels = cube.reshape(-1, 25)
        labels = label.flatten()
        
        # Filter background (0 usually?) or just take valid classes
        # Assuming 0-5 are classes.
        indices = np.random.choice(len(labels), 2000, replace=False) # More pixels
        X_list.append(pixels[indices])
        y_list.append(labels[indices])
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # SVM
    print("Fitting SVM...")
    svm = LinearSVC(C=0.01, penalty="l1", dual=False)
    svm.fit(X, y)
    svm_weights = np.sum(np.abs(svm.coef_), axis=0)
    svm_ranking = np.argsort(svm_weights)[::-1]

    # Fisher Score
    print("Calculating Fisher Score...")
    # F = sum( n_c * (mu_c - mu)^2 ) / sum( n_c * sigma_c^2 )
    mean_total = np.mean(X, axis=0)
    fisher_num = np.zeros(25)
    fisher_den = np.zeros(25)
    
    classes = np.unique(y)
    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        if n_c == 0: continue
        mu_c = np.mean(X_c, axis=0)
        sigma_c = np.var(X_c, axis=0)
        
        fisher_num += n_c * (mu_c - mean_total)**2
        fisher_den += n_c * sigma_c
        
    fisher_scores = fisher_num / (fisher_den + 1e-8)
    fisher_ranking = np.argsort(fisher_scores)[::-1]

    # Correlation (Pearson with GT)
    # We treat labels as ordinal for simple correlation or one-vs-all. 
    # Better: Average correlation with each one-hot class vector.
    print("Calculating Correlation...")
    corr_scores = np.zeros(25)
    y_onehot = pd.get_dummies(y).values
    for b in range(25):
        band_vals = X[:, b]
        # Average abs correlation with each class
        corrs = [abs(np.corrcoef(band_vals, y_onehot[:, i])[0,1]) for i in range(y_onehot.shape[1])]
        corr_scores[b] = np.mean(corrs)
    
    corr_ranking = np.argsort(corr_scores)[::-1]

    # Output results
    results = {
        "Entropy": entropy_scores,
        "Variance": var_scores,
        "SVM": svm_weights,
        "Fisher": fisher_scores,
        "Correlation": corr_scores
    }
    
    print("\n--- Rankings (Indices 0-24) ---")
    print(f"Entropy: {list(entropy_ranking)}")
    print(f"Variance: {list(var_ranking)}")
    print(f"SVM-RFE: {list(svm_ranking)}")
    print(f"Fisher: {list(fisher_ranking)}")
    print(f"Correlation: {list(corr_ranking)}")
    
    # Save to file for the runner
    with open("band_rankings.txt", "w") as f:
        f.write(f"Entropy:{','.join(map(str, entropy_ranking))}\n")
        f.write(f"Variance:{','.join(map(str, var_ranking))}\n")
        f.write(f"SAD:{','.join(map(str, var_ranking))}\n") # Map SAD to Variance (High energy)
        f.write(f"SVM:{','.join(map(str, svm_ranking))}\n")
        f.write(f"Fisher:{','.join(map(str, fisher_ranking))}\n")
        f.write(f"Correlation:{','.join(map(str, corr_ranking))}\n")

if __name__ == "__main__":
    calculate_rankings()
