# src/models/__init__.py

# --- Import các lớp model mới ---
from .unet_cbam import UnetCbam
from .unet_seattention import UnetSEAttention
from .unet_bandselector import UnetBandS
from .unet_cbam_seattention import UnetCbamSeattention
from .unet_bcd import UnetBcd
from .unet_depthwiseseparable import UnetDepthwiseSeparable
from .unet_deepsupervision import UnetDeepSupervision

# --- Import model cũ để tương thích ngược ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from models_ignored_temp.gemini_unet_v2 import GeminiUNetV2
from models_ignored_temp.flash_unet import FlashUNet
from models_ignored_temp.gemini_unet import GeminiUNet
from models_ignored_temp.hybrid_unet import HybridUNet

def get_model(name: str, params: dict):
    """
    Model factory mới, linh hoạt.
    Tên model giờ đây tương ứng trực tiếp với tên class.
    """
    models = {
        # Model mới
        'UnetCbam': UnetCbam,
        'UnetSEAttention': UnetSEAttention,
        'UnetBandS': UnetBandS,
        'UnetCbamSeattention': UnetCbamSeattention,
        'UnetBcd': UnetBcd,
        'UnetDepthwiseSeparable': UnetDepthwiseSeparable,
        'UnetDeepSupervision': UnetDeepSupervision,

        # Model cũ để tương thích ngược
        'GeminiUNetV2': GeminiUNetV2,
        'FlashUNet': FlashUNet,
        'GeminiUNet': GeminiUNet,
        'HybridUNet': HybridUNet,
    }

    if name in models:
        return models[name](**params)
    else:
        raise ValueError(f"Model '{name}' not recognized. Available models are: {list(models.keys())}")
    
    

