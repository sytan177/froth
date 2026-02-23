import torch
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any
from .registry import ModelRegistry

@dataclass
class SnapConfig:
    normalize: bool = True
    z_bin_size: float = 0.05
    R_bin_size: float = 2.0
    save_scaler: bool = False
    n_jobs: int = 6
    keep_cache: bool = False
    force_recompute: bool = False
    
@dataclass
class ImageConfig:
    rotate: bool = False
    background: bool = False
    background_val: float = 1.0
    smoothing_window: int = 30
    median_window: int = 20
    stats_sigma: int = 5
    opening_iteration: int = 10
    save_scaler: bool = False
    small_hole_size: int = 1800
    batch_size: int = 500000

@dataclass
class ModelConfig:
    model_name: str
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    processing_config: Optional[Union[SnapConfig, ImageConfig]] = None
    _registry: Optional[ModelRegistry] = field(default=None, init=False, repr=False)
    _config: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        from .registry import ModelRegistry
        self._registry = ModelRegistry()
        self._config = self._registry.get_model_config(self.model_name)
        
    def get_required_features(self):
        return self._config.get('input_features', ['density', 'temperature'])
    
    def load_model(self):
        device = torch.device(self.device)
        model_class = self._get_model_class(self._config['model_class'])
        checkpoint_path = self._registry.get_model_path(self.model_name)
        arch_config = self._config['architecture']
        model = model_class(**arch_config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    def _get_model_class(self, class_name):
        from . import models
        if not hasattr(models, class_name):
            raise ValueError(
                f"   ⚠   Unknown model class: {class_name} \n"
                f"   ·   : {[name for name in dir(models) if not name.startswith('_')]}")
        return getattr(models, class_name)