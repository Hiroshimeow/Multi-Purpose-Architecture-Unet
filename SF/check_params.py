
import sys
import os
import torch
import yaml
from pathlib import Path

# Add the current directory to sys.path to resolve imports correctly
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

# Mocking the imports since we are running this as a standalone script inside SF/
from models.spectral_former_seg import SpectralFormer_Seg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_config(name, params):
    print(f"--- Testing Config: {name} ---")
    try:
        model = SpectralFormer_Seg(**params)
        count = count_parameters(model)
        print(f"Parameters: {count:,} ({count/1e6:.2f} M)")
        return count
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    # Baseline configuration (from sf_base_p8.yaml but reduced)
    # Target: 0.3M - 0.5M
    
    # Attempt 1: Very aggressive reduction
    params_1 = {
        'image_size': 224,
        'patch_size': 8,
        'in_channels': 5, # Testing with k=5 bands as an average case
        'num_classes': 5,
        'dim': 32,          # Reduced from 128
        'depth': 3,         # Reduced from 6
        'heads': 4,
        'mlp_dim': 64,      # 2x dim
        'dim_head': 8,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }
    
    # Attempt 2: Slightly larger
    params_2 = {
        'image_size': 224,
        'patch_size': 8,
        'in_channels': 5,
        'num_classes': 5,
        'dim': 48,
        'depth': 3,
        'heads': 4,
        'mlp_dim': 96,
        'dim_head': 12,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }

    # Attempt 3: Depth focused
    params_3 = {
        'image_size': 224,
        'patch_size': 8,
        'in_channels': 5,
        'num_classes': 5,
        'dim': 32,
        'depth': 4,        # Deeper
        'heads': 4,
        'mlp_dim': 64,
        'dim_head': 8,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }
    
    # Attempt 4: Width focused (Base Tiny candidate)
    params_4 = {
        'image_size': 224,
        'patch_size': 8,
        'in_channels': 5, 
        'num_classes': 5,
        'dim': 64,         
        'depth': 3,        
        'heads': 4,
        'mlp_dim': 128,    
        'dim_head': 16,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }

    test_config("Attempt 1 (Dim 32, Depth 3)", params_1)
    test_config("Attempt 2 (Dim 48, Depth 3)", params_2)
    test_config("Attempt 3 (Dim 32, Depth 4)", params_3)
    # Attempt 5: Patch Size 16 (Drastic reduction expected due to CAF)
    params_5 = {
        'image_size': 224,
        'patch_size': 16, # Increased from 8
        'in_channels': 5, 
        'num_classes': 5,
        'dim': 64,         
        'depth': 3,        
        'heads': 4,
        'mlp_dim': 128,    
        'dim_head': 16,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }

    # Attempt 6: Patch Size 16, slightly deeper
    params_6 = {
        'image_size': 224,
        'patch_size': 16,
        'in_channels': 5, 
        'num_classes': 5,
        'dim': 64,         
        'depth': 4,        
        'heads': 4,
        'mlp_dim': 128,    
        'dim_head': 16,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }

    test_config("Attempt 5 (P16, Dim 64, Depth 3)", params_5)
    test_config("Attempt 6 (P16, Dim 64, Depth 4)", params_6)

if __name__ == "__main__":
    main()
