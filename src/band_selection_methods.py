import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F

def entropy_selection(data, top_k=5):
    """
    Selects bands with the highest spectral information based on entropy.
    """
    scores = []
    # Assuming data shape is (height, width, bands)
    for i in range(data.shape[-1]):
        # Flatten the band to a 1D array for histogram calculation
        band_data = data[:, :, i].flatten()
        hist, _ = np.histogram(band_data, bins=256, density=True)
        # Add a small epsilon to avoid log(0)
        hist += 1e-12
        scores.append(entropy(hist))
    
    # Select top_k bands with the highest entropy scores
    selected_indices = np.argsort(scores)[-top_k:]
    print(f"Entropy-based selected band indices: {selected_indices}")
    return selected_indices

def correlation_clustering_selection(data, n_clusters=5):
    """
    Avoids redundant bands by grouping highly correlated ones and selecting a representative from each cluster.
    """
    # Reshape data to (pixels, bands)
    reshaped_data = data.reshape(-1, data.shape[-1])
    
    # Calculate the correlation matrix between bands
    corr_matrix = np.corrcoef(reshaped_data, rowvar=False)
    
    # Cluster the bands based on the correlation matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(corr_matrix)
    
    selected_indices = []
    for i in range(n_clusters):
        # Find all bands belonging to the current cluster
        cluster_bands = np.where(kmeans.labels_ == i)[0]
        if len(cluster_bands) > 0:
            # Select the first band in the cluster as the representative
            selected_indices.append(cluster_bands[0])
            
    print(f"Correlation+Clustering selected band indices: {selected_indices}")
    return selected_indices

def fisher_score_selection(X, y, k=5):
    """
    Selects bands that maximize inter-class separability using Fisher Score.
    Note: Requires flattened data (X, y).
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    print(f"Fisher Score selected band indices: {selected_indices}")
    return selected_indices

def svm_rfe_selection(X, y, k=5):
    """
    Uses a classifier (SVM) to rank and recursively eliminate less useful bands.
    Note: Requires flattened data (X, y). This can be computationally expensive.
    """
    # Using a linear SVC for efficiency
    svc = SVC(kernel="linear", C=1)
    # RFE with a step of 1 to remove one feature at a time
    selector = RFE(estimator=svc, n_features_to_select=k, step=1)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    print(f"SVM-RFE selected band indices: {selected_indices}")
    return selected_indices

def spectral_angle(a, b):
    """
    Computes the spectral angle between two vectors.
    """
    # Add epsilon to avoid division by zero
    norm_a = np.linalg.norm(a) + 1e-9
    norm_b = np.linalg.norm(b) + 1e-9
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    # Clip to handle potential floating point inaccuracies
    return np.arccos(np.clip(cos_sim, -1.0, 1.0))

def sad_selection(data, top_k=5):
    """
    Selects bands with unique spectral shapes based on Spectral Angle Distance.
    """
    num_bands = data.shape[-1]
    sad_scores = np.zeros(num_bands)
    
    # Flatten each band into a 1D vector
    flattened_bands = [data[:, :, i].flatten() for i in range(num_bands)]
    
    for i in range(num_bands):
        ref_band = flattened_bands[i]
        # Calculate the mean SAD to all other bands
        mean_sad = np.mean([spectral_angle(ref_band, flattened_bands[j]) 
                           for j in range(num_bands) if i != j])
        sad_scores[i] = mean_sad
        
    # Select top_k bands with the highest (most dissimilar) SAD scores
    selected_indices = np.argsort(sad_scores)[-top_k:]
    print(f"SAD-based selected band indices: {selected_indices}")
    return selected_indices

# ============================================================
# Learnable Band Selector (Deep Learning)
# ============================================================

class LearnableBandSelector(nn.Module):
    """
    Selects the top-k bands from the input using a learnable Gumbel-Softmax mechanism.
    This is a "hard" selection mechanism that is differentiable.
    """
    def __init__(self, in_channels: int, num_select_bands: int, temperature: float = 1.0):
        super(LearnableBandSelector, self).__init__()
        if num_select_bands > in_channels:
            raise ValueError(f"num_select_bands ({num_select_bands}) cannot be greater than in_channels ({in_channels}).")
        self.in_channels = in_channels
        self.num_select_bands = num_select_bands
        self.temperature = temperature
        
        # Learnable logits for each band. Higher logit means higher probability of being selected.
        self.logits = nn.Parameter(torch.randn(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        
        if self.training:
            # Use Gumbel-TopK trick for differentiable hard selection
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
            _, top_k_indices = torch.topk(self.logits + gumbel_noise, self.num_select_bands, dim=-1)
        else:
            # During inference, just pick the top-k bands based on the learned logits
            _, top_k_indices = torch.topk(self.logits, self.num_select_bands, dim=-1)

        # Gather the channels based on the selected indices
        output = torch.index_select(x, dim=1, index=top_k_indices)
        
        return output