# src/models/__init__.py

# Import the new GeminiUNet model
from .gemini_unet import GeminiUNet
from .hybrid_unet import HybridUNet
from .gemini_unet_v2 import GeminiUNetV2
from .flash_unet import FlashUNet

def get_model(name: str, params: dict):
    """
    Model factory.
    """
    if name == 'GeminiUNet':
        # Pass the entire params dictionary to the model
        return GeminiUNet(**params)
    elif name == 'HybridUNet':
        return HybridUNet(**params)
    elif name == 'GeminiUNetV2':
        return GeminiUNetV2(**params)
    elif name == 'FlashUNet':
        return FlashUNet(**params)
    else:
        raise ValueError(f"Model '{name}' not recognized.")
