# src/models/__init__.py

# --- Import các lớp model mới ---
from .unet_cbam import UnetCbam
from .unet_seattention import UnetSEAttention
from .unet_bandselector import UnetBandS
from .unet_cbam_seattention import UnetCbamSeattention
from .unet_bcd import UnetBcd
from .unet_depthwiseseparable import UnetDepthwiseSeparable
from .unet_deepsupervision import UnetDeepSupervision
from .asran_network import ASRAN
from .asran_lbs import ASRAN_LBS
from .vit_unet import ViT_UNet
from .unet_base import UNetBase
from .unet_prunable import UnetPrunable
from .cbsfnet import CB_SFNet # Added for pruning experiment
from .cbsfnet_unified import CB_SFNet_Unified
from .cbsfnet_ag import CB_SFNet_AG
from .gemini_unet_v2 import GeminiUNetV2
from .simple_unet import SimpleUNet

# --- Import model cũ để tương thích ngược (ĐÃ BỊ XÓA VÌ GÂY LỖI) ---
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.parent))
# from models_ignored_temp.gemini_unet_v2 import GeminiUNetV2
# from models_ignored_temp.flash_unet import FlashUNet
# from models_ignored_temp.gemini_unet import GeminiUNet
# from models_ignored_temp.hybrid_unet import HybridUNet

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
        'ASRAN': ASRAN,
        'CB_SFNet': CB_SFNet,
        'CB_SFNet_Unified': CB_SFNet_Unified,
        'CB_SFNet_AG': CB_SFNet_AG,
        'ASRAN_LBS': ASRAN_LBS,
        'ViT_UNet': ViT_UNet,
        'UnetPrunable': UnetPrunable,
        'GeminiUNetV2': GeminiUNetV2,
        'SimpleUNet': SimpleUNet,

        # Model cũ để tương thích ngược (ĐÃ BỊ XÓA VÌ GÂY LỖI)
        # 'FlashUNet': FlashUNet,
        # 'GeminiUNet': GeminiUNet,
        # 'HybridUNet': HybridUNet,
    }

    if name in models:
        # If a nested 'params' key exists, use it for the model's kwargs.
        # Otherwise, fall back to the old behavior for compatibility with other models.
        if 'params' in params:
            model_params = params['params']
        else:
            model_params = {k: v for k, v in params.items() if k not in ['name', 'class_names']}
        return models[name](**model_params)
    else:
        raise ValueError(f"Model '{name}' not recognized. Available models are: {list(models.keys())}")
