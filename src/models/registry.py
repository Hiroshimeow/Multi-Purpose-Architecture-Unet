"""
Model Registry Pattern - Quản lý động các model architectures
"""
import torch.nn as nn
from typing import Dict, Callable, Any
import importlib


class ModelRegistry:
    """Registry để đăng ký và truy xuất models động"""
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """Decorator để đăng ký model"""
        def decorator(model_class):
            nonlocal name
            if name is None:
                name = model_class.__name__
            cls._registry[name.lower()] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str, **kwargs) -> nn.Module:
        """Lấy model từ registry"""
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_models(cls):
        """Liệt kê tất cả models đã đăng ký"""
        return list(cls._registry.keys())


# Đăng ký models
@ModelRegistry.register('standardunet')
class StandardUNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_filters=16, 
                 use_asa=True, use_sr_head=False, sr_out_channels=25):
        super().__init__()
        # ... implementation từ model_def.py
        pass