import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary

# Import your model and BandSelector
from src.models.gemini_unet_v2 import GeminiUNetV2
from Band_Selector import BandSelector

def calculate_stats(model_name, model_params, input_size=(1, 25, 224, 224), device='cpu'):
    print(f"\n--- Calculating stats for {model_name} ---")
    
    # Adjust input_size based on whether BandSelector is used
    if 'band_selector_config' in model_params and model_params['band_selector_config'] is not None:
        # If BandSelector is used, the actual input to the model is 25 channels
        # The model's in_channels param will be the output of BandSelector (e.g., 5)
        # So, we need to pass the original 25 channels to the model for FLOPs/Params calculation
        # and ensure the model's internal logic handles it.
        # The input_size should reflect the actual input to the model (25 channels)
        pass # input_size is already (1, 25, 224, 224)
    else:
        # If BandSelector is NOT used, the model's in_channels is the actual input channels
        input_size = (1, model_params['in_channels'], input_size[2], input_size[3])

    model = GeminiUNetV2(**model_params).to(device)
    dummy_input = torch.randn(input_size).to(device)

    # Calculate FLOPs
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2 / (10**9) # Convert MACs to GFLOPs (MACs * 2 for FLOPs, /10^9 for G) 
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / (10**6) # Convert to M

    print(f"FLOPs (G): {flops:.4f}")
    print(f"Params (M): {total_params:.4f}")
    return flops, total_params

if __name__ == '__main__':
    device = 'cpu' # Use CPU for stats calculation to avoid GPU memory issues

    # --- Model Configurations ---
    # UNet-base (estimated: Run 3 without SAC, bilinear=True)
    unet_base_params = {
        'in_channels': 25,
        'num_classes': 6,
        'initial_filters': 32,
        'depth': 4,
        'dropout_rate': 0.2,
        'bilinear': True,
        'sac_params': None,
        'use_bcd': False,
        'band_selector_config': None
    }
    calculate_stats("UNet-base", unet_base_params)

    # run6 (Large model with data augmentation)
    run6_params = {
        'in_channels': 25,
        'num_classes': 6,
        'initial_filters': 64,
        'depth': 4,
        'dropout_rate': 0.2,
        'bilinear': False,
        'sac_params': {'input_depth': 25, 'reduced_depth': 3, 'channels': 64, 'use_sea': True},
        'use_bcd': False,
        'band_selector_config': None
    }
    calculate_stats("run6", run6_params)

    # run8 (Small model with Transposed Convolutions)
    run8_params = {
        'in_channels': 25,
        'num_classes': 6,
        'initial_filters': 32,
        'depth': 4,
        'dropout_rate': 0.2,
        'bilinear': False,
        'sac_params': {'input_depth': 25, 'reduced_depth': 3, 'channels': 32, 'use_sea': True},
        'use_bcd': False,
        'band_selector_config': None
    }
    calculate_stats("run8", run8_params)

    # run8_300 (Depth 4, Band Selector)
    run8_300_params = {
        'in_channels': 5, # Model expects 5 channels after BandSelector
        'num_classes': 6,
        'initial_filters': 32,
        'depth': 4,
        'dropout_rate': 0.2,
        'bilinear': False,
        'sac_params': {'input_depth': 5, 'reduced_depth': 3, 'channels': 32, 'use_sea': True},
        'use_bcd': False,
        'band_selector_config': {'original_in_channels': 25, 'num_selected_bands': 5}
    }
    calculate_stats("run8_300", run8_300_params)

    # run8_300_dep5 (Depth 5, Band Selector)
    run8_300_dep5_params = {
        'in_channels': 5, # Model expects 5 channels after BandSelector
        'num_classes': 6,
        'initial_filters': 32,
        'depth': 5, # Increased depth
        'dropout_rate': 0.2,
        'bilinear': False,
        'sac_params': {'input_depth': 5, 'reduced_depth': 3, 'channels': 32, 'use_sea': True},
        'use_bcd': False,
        'band_selector_config': {'original_in_channels': 25, 'num_selected_bands': 5}
    }
    calculate_stats("run8_300_dep5", run8_300_dep5_params)
