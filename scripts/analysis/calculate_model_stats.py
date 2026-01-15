import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary
import yaml
import importlib
import re

def calculate_stats_from_config(config_path, device='cpu'):
    print(f"\n--- Calculating stats for model from config: {config_path} ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    model_params = config['model']['params']
    data_dir = config['data']['dir']

    # Dynamically import the model
    if model_name == "ASRAN":
        module = importlib.import_module("src.models.asran_network")
        model_class = getattr(module, "ASRAN")
    elif model_name == "CB_SFNet":
        module = importlib.import_module("src.models.cbsfnet")
        model_class = getattr(module, "CB_SFNet")
    elif model_name == "GeminiUNetV2":
        module = importlib.import_module("src.models.gemini_unet_v2")
        model_class = getattr(module, "GeminiUNetV2")
    else:
        try:
            # Convert CamelCase to snake_case for filename, e.g., GeminiUNetV2 -> gemini_unet_v2
            module_name_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', model_name).lower()
            module = importlib.import_module(f"src.models.{module_name_snake}")
            model_class = getattr(module, model_name)
        except (ImportError, AttributeError):
            raise ValueError(f"Model {model_name} not found. Attempted to load from 'src.models.{module_name_snake}'. Check filename and class name.")

    # Determine input_channels based on data_dir
    if "pca_3band" in data_dir:
        input_channels = 3
    elif "pca_5band" in data_dir:
        input_channels = 5
    else: # Default to 25 for full Image_dataset
        input_channels = 25 
    
    # Override in_channels in model_params if it exists
    model_params['in_channels'] = input_channels

    # Create a dummy input based on the config
    patch_size = config.get('data', {}).get('patching', {}).get('patch_size', 224)
    input_size = (1, input_channels, patch_size, patch_size)

    model = model_class(**model_params).to(device)
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
    
    import argparse
    parser = argparse.ArgumentParser(description='Calculate model stats from a config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration YAML file.')
    args = parser.parse_args()

    calculate_stats_from_config(args.config, device)

