
import torch
from src.models.unet_base import UNetBase

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Configuration from the paper/request
# base_filters in config corresponds to initial_filters in the model
config = {
    'in_channels': 5,  # Number of bands after selection (k=5 is a good representative value)
    'num_classes': 5,  # As seen in previous configs
    'initial_filters': 6, # Changed back to 6 for verification
    'depth': 4,
    'bilinear': True
}

# Instantiate the model
model = UNetBase(**config)

# Calculate and print the number of parameters
params_M = count_parameters(model) / 1_000_000
print(f"Model: UNetBase")
print(f"Config: initial_filters={config['initial_filters']}, depth={config['depth']}")
print(f"Total Trainable Params: {params_M:.3f}M")

# Check against the target
target_params = 0.17 # Target now 0.17M as per GEMINI.md
if abs(params_M - target_params) < 0.02:
    print(f"SUCCESS: Parameter count ({params_M:.3f}M) is very close to the target (~{target_params}M).")
else:
    print(f"INFO: Parameter count ({params_M:.3f}M) is not the target (~{target_params}M). Please check the architecture.")
