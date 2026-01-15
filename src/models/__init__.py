from .unet_bandselector import UnetBandS as OriginalUnetBandS
from .asran_network import ASRAN
from .unet_base import UNetBase

class StandardUNet(UNetBase):
    """Wrapper to ensure StandardUNet can be called with legacy kwargs it doesn't use."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TABS(OriginalUnetBandS):
    """Wrapper to ensure TABS can be called with legacy kwargs it doesn't use."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def get_model(name: str, params: dict):
    """
    Model factory.
    """
    models = {
        'TABS': TABS,
        'ASRAN_LBS': TABS,
        'StandardUNet': StandardUNet,
        'ASRAN': ASRAN,
    }

    if name in models:
        if 'params' in params:
            model_params = params['params'].copy() # Use .copy() to avoid modifying original config dict directly
        else:
            model_params = {k: v for k, v in params.items() if k not in ['name', 'class_names']}

        # Map 'base_filters' from config to 'initial_filters' for UNetBase/StandardUNet
        if 'base_filters' in model_params:
            model_params['initial_filters'] = model_params.pop('base_filters')
        
        return models[name](**model_params)
    else:
        raise ValueError(f"Model '{name}' not recognized. Available models are: {list(models.keys())}")