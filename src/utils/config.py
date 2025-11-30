"""
Simple argument parser for notebook usage
"""

import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any


class ParserArgs:
    """Simple class to hold training arguments for notebook usage"""
    
    def __init__(self, config_path: str = None, **kwargs):
        """
        Initialize arguments from YAML config and/or kwargs
        
        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        **kwargs : dict
            Additional arguments to override config
        """
        # Load from YAML if provided
        if config_path is not None:
            config_path = Path(config_path).expanduser()
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Update with kwargs
        config.update(kwargs)
        
        # Set attributes
        for key, value in config.items():
            if key == 'model_config' and isinstance(value, dict):
                # Handle nested model config
                for mk, mv in value.items():
                    setattr(self, mk, mv)
            else:
                setattr(self, key, value)
        
        # Process special attributes
        self._process_attributes()
    
    def _process_attributes(self):
        """Process and validate attributes"""
        # Handle device
        if hasattr(self, 'device'):
            if self.device == 'cuda' and not torch.cuda.is_available():
                print("CUDA not available, using CPU")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle dtype
        if hasattr(self, 'dtype'):
            if self.dtype == 'float32':
                self.dtype = torch.float32
            elif self.dtype == 'float64':
                self.dtype = torch.float64
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32
        
        # Expand paths
        if hasattr(self, 'dataset_path'):
            self.dataset_path = str(Path(self.dataset_path).expanduser())
        if hasattr(self, 'storage_path'):
            self.storage_path = str(Path(self.storage_path).expanduser())
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"ParserArgs({attrs})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def load_config(config_path: str) -> ParserArgs:
    """
    Load configuration from YAML file
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    
    Returns
    -------
    ParserArgs
        Arguments object
    """
    return ParserArgs(config_path)



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train a model on a dataset using unified interfaces',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    return parser.parse_args()