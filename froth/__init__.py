from pathlib import Path
import shutil

from .loader import SnapData, ImageData
from .config import ImageConfig, SnapConfig
from .aerate import BubbleClassifier
from .knock import BubbleExtractor
from .utils import setup_logging

import logging
logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__author__ = "Shuyu Tan"
__email__ = "lw185@uni-heidelberg.de"
__license__ = "MIT"
__description__ = "Superbubble identification with graph neural networks"

def create_default_config(output_path="config.yaml", overwrite:bool =False, verbose: bool = True):   
    setup_logging(verbose)
    template_path = Path(__file__).parent / "config.default.yaml"
    output = Path(output_path)
    
    if not template_path.exists():
        raise FileNotFoundError(
            f"   ⚠   Template config not found in package: {template_path}\n"
            f"   ⚠   This may indicate a corrupted installation."
        )
    
    if output.exists() and not overwrite:
        user_input = input(
            f"-----> File '{output}' already exists. Overwrite? [y/N]: "
        ).strip().lower()
        
        if user_input not in ('y', 'yes'):
            print("   ⚠   Cancelled. Existing file not modified.")
            return output
    
    shutil.copy(template_path, output)
    
    logger.info(f"   ✔   Created {output}")
    logger.info("   ·   Edit this file with your data paths before running froth.")
    logger.info(f"       example: snap_data = SnapData(snap_num=500, config_file='{output}')")
    return output

def get_template_path():
    return Path(__file__).parent / "config.default.yaml"
