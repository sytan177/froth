import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import copy

class ConfigLoader:
    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = Path(__file__).parent / 'data' / 'configs'
        self.config_dir = Path(config_dir)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        full_path = self.config_dir / config_path
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for base config
        if 'base_config' in config:
            base_config_path = config.pop('base_config')
            base_config = self.load_config(base_config_path)  # Recursive
            
            # Deep merge
            merged = self._deep_merge(base_config, config)
            return merged
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class ModelRegistry:    
    def __init__(self, registry_path=None):
        if registry_path is None:
            registry_path = Path(__file__).parent / 'data' / 'model_registry.yaml'
        
        self.registry_path = Path(registry_path)
        
        with open(self.registry_path, 'r') as f:
            self.registry = yaml.safe_load(f)
        
        self.config_loader = ConfigLoader()
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        
        if model_name not in self.registry['models']:
            available = list(self.registry['models'].keys())
            raise ValueError(
                f"Unknown model: {model_name}\n"
                f"Available models: {available}"
            )
        
        model_info = self.registry['models'][model_name]
        config_path = model_info['config']
        
        config = self.config_loader.load_config(config_path)
        config['model_name'] = model_name
        config['tags'] = model_info.get('tags', [])
        config['registry_info'] = model_info
        return config
    
    def list_models(self, tags: Optional[List[str]] = None, 
                   model_type: Optional[str] = None) -> Dict[str, Dict]:
        models = self.registry['models']
        if tags:
            models = {
                name: info for name, info in models.items()
                if any(tag in info.get('tags', []) for tag in tags)}
        if model_type:
            models = {
                name: info for name, info in models.items()
                if model_type in name.lower()}
        return models
    
    def get_model_path(self, model_name: str) -> Path:
        config = self.get_model_config(model_name)
        checkpoint_rel = config['checkpoint']
        
        data_dir = Path(__file__).parent / 'data'
        return data_dir / checkpoint_rel