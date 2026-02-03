from .spectral_former_seg import SpectralFormer_Seg

def get_model(config):
    model_name = config['model']['name']
    params = config['model']['params']
    
    if model_name == 'SpectralFormer_Seg':
        # Filter out params that are for Dataset/Config but not for the Model class
        model_params = params.copy()
        if 'selected_bands_indices' in model_params:
            del model_params['selected_bands_indices']
        return SpectralFormer_Seg(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
