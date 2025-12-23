# src/models/__init__.py

# Essential models for the experiment
from .unet_bandselector import UnetBandS as OriginalUnetBandS
# from .backbones import StandardUNet # Missing
from .asran_network import ASRAN
from .unet_base import UNetBase

class StandardUNet(UNetBase):
    def __init__(self, base_filters=64, use_sa=False, **kwargs):
        # Map base_filters to initial_filters
        super().__init__(initial_filters=base_filters, **kwargs)

class ASRAN_LBS(OriginalUnetBandS):
    def __init__(self, base_filters=64, **kwargs):
        # Map base_filters to initial_filters for the inner UNet
        super().__init__(initial_filters=base_filters, **kwargs)

def get_model(name: str, params: dict):
    """
    Model factory.
    """
    models = {
        'ASRAN_LBS': ASRAN_LBS,
        'StandardUNet': StandardUNet,
        'ASRAN': ASRAN,
    }

    if name in models:
        if 'params' in params:
            model_params = params['params']
        else:
            model_params = {k: v for k, v in params.items() if k not in ['name', 'class_names']}
        return models[name](**model_params)
    else:
        raise ValueError(f"Model '{name}' not recognized. Available models are: {list(models.keys())}")